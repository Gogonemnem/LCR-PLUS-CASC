"""Single entry point for the CASC (BERT) and LCR (RoTHop++) pipelines.

Subcommands
-----------
  prep   Build vocabularies / extract spans / score / generate labels (CASC preprocessing).
  casc   Train and/or evaluate the BERTLinear CASC model.
  lcr    Train / tune / evaluate the two-tower LCRRothopPP model.

A named config recipe selects the hyperparameters (defaults: `casc` ->
configs/casc.yaml, `lcr` -> configs/lcr.yaml); pick another with
--config <name> (configs/<name>.yaml). Each training run writes artifacts
to output/<config-name>/<timestamp>_<kind>/: config.json, run.log,
predictions.csv, plus casc_model.pth / casc_checkpoint.pth (CASC) or
lcr_model.pth / lcr_checkpoint.pth (LCR).

Devices
-------
By default every torch module is wrapped with DataParallel and fans a batch
out across ALL visible GPUs (labeling scoring and training alike). Pass
--device cuda:N to pin a single GPU, or --device cpu for CPU.

The phases run are implied by which data paths are given: --train-path trains,
--val-path adds a validation split, --test-path evaluates. Give both a train and
a test path to train then evaluate. To evaluate only, give --test-path (and
--load to pick the checkpoint; otherwise the newest trained artifact is used).

Resume a crashed run (after N more epochs are added automatically) with:
  python main.py casc --resume                    # newest checkpoint across all runs
  python main.py casc --resume <output-casc_run-dir>      # a specific run dir
  python main.py casc --resume <path/to/casc_checkpoint.pth> # a specific checkpoint file

Examples
--------
  # CASC train+eval, or LCR train+eval (both GPUs via DataParallel by default)
  python main.py casc --train-path <train.txt> --test-path <test.txt>
  python main.py lcr --train-path <train.txt> --test-path <test.txt>

  # run both models separately (two runs), one per GPU if desired
  python main.py --device cuda:0 casc --train-path <train.txt> &
  python main.py --device cuda:1 lcr --train-path <train.txt> & wait

  # LCR hyperparameter search
  python main.py lcr --tune --train-path <train.txt>

  # evaluate an existing checkpoint only
  python main.py casc --test-path <test.txt> --load <path/to/casc_model.pth>

  # resume the LCR phase from a specific checkpoint
  python main.py lcr --train-path <train.txt> --resume <path/to/lcr_checkpoint.pth>
"""
import argparse
import logging
import os


def _config(args):
    """Apply --domain/--device/--gce-q/CLI overrides to the package config before importing data modules."""
    import lcr_plus_casc.config as cfg
    if args.domain:
        cfg.config['domain'] = args.domain
    if args.device:
        cfg.config['device'] = args.device
    if getattr(args, 'gce_q', None) is not None:
        cfg.config['gce_q'] = args.gce_q
    if getattr(args, 'seed', None) is not None:
        cfg.training.seed = args.seed
    if getattr(args, 'max_length', None) is not None:
        cfg.config['max_length'] = args.max_length
    return cfg


def _pin_env(device):
    """Set CUDA_VISIBLE_DEVICES for `--device cuda:N` / `cpu` (must run before any CUDA init).

    Bare `cuda` is left untouched so every visible GPU stays available for
    DataParallel.
    """
    if not device:
        return
    if device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        return
    if device.startswith('cuda') and ':' in device:
        os.environ['CUDA_VISIBLE_DEVICES'] = device.rsplit(':', 1)[-1]


def _pin_device(args):
    """Resolve the process device before torch creates a CUDA context.

    --device cuda:N -> CUDA_VISIBLE_DEVICES=N (the process sees one GPU as
    cuda:0); --device cuda (or omitted) -> all visible GPUs, config device
    'cuda' (models are DataParallel-wrapped); --device cpu -> CPU.
    """
    _pin_env(args.device)
    if not args.device:
        # Recipe/default already chose 'cuda' when GPUs exist; log the fan-out.
        import lcr_plus_casc.config as cfg

        if cfg.config['device'] == 'cuda':
            import torch

            n = torch.cuda.device_count()
            if n > 1:
                logging.getLogger(__name__).info(
                    'device cuda: DataParallel across %d GPUs', n)
        return
    import lcr_plus_casc.config as cfg

    if args.device == 'cpu':
        cfg.config['device'] = 'cpu'
        return
    if args.device.startswith('cuda'):
        if ':' in args.device:
            idx = args.device.rsplit(':', 1)[-1]
            logging.getLogger(__name__).info(
                'pinned to physical GPU %s (CUDA_VISIBLE_DEVICES=%s) -> device cuda:0',
                idx, idx)
            cfg.config['device'] = 'cuda:0'
            args.device = 'cuda:0'
        else:
            cfg.config['device'] = 'cuda'


def _resume_checkpoint_names(kind):
    from lcr_plus_casc.config import CKPT

    if kind == 'lcr_tune':
        return [CKPT['lcr']]
    return [CKPT[kind]]


def _load_recipe(name):
    import lcr_plus_casc.config as cfg

    cfg.load_config_file(name)
    logging.getLogger(__name__).info('config recipe -> %s', cfg.config['config_file'])


def _resolve_resume(resume, kind):
    """Resolve --resume to a checkpoint file path.

    Accepts the checkpoint file itself, a run dir containing one, or bare
    --resume (the newest checkpoint under output_root, across all recipes).
    """
    import glob
    from lcr_plus_casc import runs
    from lcr_plus_casc.config import config

    if not resume:
        return None
    names = _resume_checkpoint_names(kind)

    def check(path):
        if os.path.basename(path) not in names:
            raise SystemExit(f'--resume checkpoint {path} is not one of: {", ".join(names)}')
        logging.getLogger(__name__).info('resuming from checkpoint %s', path)
        return path

    def dir_ckpt(directory):
        found = [os.path.join(directory, n) for n in names if os.path.isfile(os.path.join(directory, n))]
        return check(found[0]) if found else None

    if isinstance(resume, str):
        candidates = [resume]
        root = runs.artifact_root()
        if root:
            candidates.append(os.path.join(root, resume))
        candidates.append(os.path.join(config['output_root'], resume))
        for cand in candidates:
            if cand.endswith('.pth') and os.path.isfile(cand):
                return check(cand)
            if os.path.isdir(cand):
                found = dir_ckpt(cand)
                if found:
                    return found
        raise SystemExit(f'--resume path not found: {resume}')
    # Bare --resume: the newest checkpoint across all recipe dirs (dedicated
    # <kind> runs plus shared `all` runs, where checkpoints land in both).
    suffixes = [f'_{kind}', '_all'] if kind in ('casc', 'lcr') else [f'_{kind}']
    matches = []
    for suffix in suffixes:
        for n in names:
            matches.extend(glob.glob(os.path.join(config['output_root'], '*', f'*{suffix}', n)))
    return check(sorted(matches)[-1]) if matches else None


# ------------------------------- preprocessing --------------------------------
def cmd_prep(args):
    _config(args)
    from lcr_plus_casc.labeling import run_labeling

    run_labeling()
    print('prep complete')


def _ensure_run(args, kind, extra=None):
    """Pick the run dir: --resume checkpoint dir > dir already started this invocation > new.

    Sets config['run_dir'] and wires the run.log handlers; returns the run dir.
    When --resume is given the resolved checkpoint path is stashed on args as
    args._resume_ckpt for the trainer to pass to train_model(resume=...).
    """
    import lcr_plus_casc.config as cfg
    from lcr_plus_casc import runs

    if getattr(args, 'resume', None):
        resume_ckpt = _resolve_resume(args.resume, kind)
        if not resume_ckpt:
            raise SystemExit(f'No runnable checkpoint ({", ".join(_resume_checkpoint_names(kind))}) found to resume ({kind})')
    else:
        resume_ckpt = None
    args._resume_ckpt = resume_ckpt
    if resume_ckpt:
        run_dir = os.path.dirname(resume_ckpt)
    else:
        existing = cfg.config.get('run_dir')
        if existing and os.path.isdir(existing) and os.path.isfile(os.path.join(existing, 'config.json')):
            run_dir = existing
        else:
            run_dir = runs.start_run(kind, extra=extra)

    cfg.config['run_dir'] = run_dir
    runs.setup_logging(run_dir)
    logging.getLogger(__name__).info('run artifacts -> %s', run_dir)
    return run_dir


def _resolve_load(run_dir, filename, suffixes):
    """Model path for eval: this run dir first, else the newest trained run."""
    from lcr_plus_casc import runs

    local = os.path.join(run_dir, f'{filename}.pth')
    if os.path.isfile(local):
        return local
    return runs.latest_artifact(f'{filename}.pth', suffixes) or filename


# ----------------------------------- CASC -------------------------------------
def cmd_casc(args):
    logger = logging.getLogger(__name__)
    _config(args)
    from lcr_plus_casc import data
    from lcr_plus_casc.classifiers.training import CASC

    if not args.train_path and not args.test_path:
        raise SystemExit('casc: give --train-path (to train), --test-path (to evaluate), '
                         'or both (train then evaluate)')

    run_dir = _ensure_run(args, 'casc', extra={'epochs': args.epochs, 'device': args.device})

    trainer = CASC()
    if args.train_path:
        sent_tr, cat_tr, pol_tr = data.load_training_data(training_path=args.train_path)
        if args.val_path:
            sent_va, cat_va, pol_va = data.load_training_data(training_path=args.val_path)
        else:
            sent_va = cat_va = pol_va = None
            logger.info('casc: no --val-path; training without validation (no early stopping, '
                        'the model is saved every epoch)')
        trainer.train_model(sent_tr, cat_tr, pol_tr, sent_va, cat_va, pol_va,
                            epochs=args.epochs, batch_size=args.batch_size,
                            resume=getattr(args, '_resume_ckpt', None))
        if args.save:
            trainer.save_model(args.save)
    if args.test_path:
        # After an in-process train, swap the best-epoch weights back in before
        # evaluating; for an eval-only run this loads the checkpoint to test.
        trainer.load_model(args.load or _resolve_load(run_dir, 'casc_model', ('_casc', '_all')))
        sent_te, cat_te, pol_te = data.load_training_data(training_path=args.test_path)
        trainer.evaluate(sent_te, cat_te, pol_te)


# ----------------------------------- LCR --------------------------------------
def _default_val_split(sent_train, cat_train, pol_train):
    """Random subset of the training rows, used by `tune` when no --val-path is given."""
    import torch
    from lcr_plus_casc.config import training

    idx = torch.randperm(len(sent_train))[:training.validation_data_size].tolist()
    return ([sent_train[i] for i in idx],
            [cat_train[i] for i in idx],
            [pol_train[i] for i in idx])


def cmd_lcr(args):
    cfg = _config(args)
    from lcr_plus_casc import data
    from lcr_plus_casc.classifiers.training import LCR

    logger = logging.getLogger(__name__)
    dcfg = cfg.domain()

    def make_trainer():
        return LCR(hidden_units=args.hidden_units, hop=args.hop,
                    drop_1=args.drop_1, drop_2=args.drop_2,
                    num_cat=len(dcfg.categories),
                    num_pol=len(dcfg.polarities))

    if not args.train_path and not args.test_path:
        raise SystemExit('lcr: give --train-path (to train), --test-path (to evaluate), '
                         'or both (train then evaluate)')

    if args.train_path:
        sent_train, cat_train, pol_train = data.load_training_data(training_path=args.train_path)
    else:
        sent_train = cat_train = pol_train = None
    if args.val_path:
        sent_val, cat_val, pol_val = data.load_training_data(training_path=args.val_path)
    else:
        sent_val = cat_val = pol_val = None

    if args.tune and not args.train_path:
        raise SystemExit('lcr --tune requires --train-path')

    if args.tune:
        if sent_val is None:
            logger.info('lcr tune: no --val-path; using a random validation split of the '
                        'training rows (validation_data_size=%d)',
                        cfg.training.validation_data_size)
            sent_val, cat_val, pol_val = _default_val_split(sent_train, cat_train, pol_train)
        _ensure_run(args, 'lcr_tune', extra={'device': args.device})
        make_trainer().tune(sent_train, cat_train, pol_train, sent_val, cat_val, pol_val,
                            max_epochs=args.epochs)
        return

    if args.train_path and sent_val is None:
        logger.info('lcr: no --val-path; training without validation (no early stopping, '
                    'the model is saved every epoch)')

    run_dir = _ensure_run(args, 'lcr', extra={'epochs': args.epochs, 'device': args.device})

    trainer = None
    if args.train_path:
        trainer = make_trainer()
        trainer.train_model(
            sent_train, cat_train, pol_train, sent_val, cat_val, pol_val,
            epochs=args.epochs, batch_size=args.batch_size,
            learning_rate=args.lr, bert_lr=args.bert_lr,
            q=args.q, l1=args.l1, l2=args.l2,
            early_stopping=not args.no_early_stop,
            resume=getattr(args, '_resume_ckpt', None))

    if args.test_path:
        # Reuse the train-time trainer when possible so a combined run builds
        # the BERT backbone once; load_model swaps in the best-epoch weights
        # in place (or loads the standalone checkpoint for an eval-only run).
        if trainer is None:
            trainer = make_trainer()
        trainer.load_model(args.load or _resolve_load(run_dir, 'lcr_model', ('_lcr', '_all')))
        sent_test, cat_test, pol_test = data.load_training_data(training_path=args.test_path)
        trainer.evaluate(sent_test, cat_test, pol_test)


# --------------------------------- CLI setup ----------------------------------
def build_parser():
    from lcr_plus_casc.config import training as cfg
    p = argparse.ArgumentParser(description='CASC + LCR aspect/polarity pipeline')
    p.add_argument('--domain', default=None, choices=['restaurant', 'laptop'])
    p.add_argument('--device', default=None, help="e.g. cuda:0 or cpu")
    p.add_argument('--seed', type=int, default=None,
                   help='random seed (default from config)')
    sub = p.add_subparsers(dest='command', required=True)

    sub.add_parser('prep', help='CASC preprocessing (vocab/extract/score/label)')

    pc = sub.add_parser('casc', help='train/eval the BERTLinear CASC model')
    pc.add_argument('--load', default=None, help='checkpoint name/path to load for eval')
    pc.add_argument('--save', default=None, help='checkpoint name to save after train')
    pc.add_argument('--resume', nargs='?', const=True, default=None,
                    metavar='RUN_DIR|CHECKPOINT',
                    help='resume from checkpoint.pth (of RUN_DIR, a CHECKPOINT file path, '
                         'or the newest run when omitted)')
    pc.add_argument('--epochs', type=int, default=cfg.epochs)
    pc.add_argument('--gce-q', type=float, default=None,
                    help='GCE loss q (default from config: 0.4)')
    pc.add_argument('--max-length', type=int, default=None,
                    help='max token length for tokenization (default from config)')
    pc.add_argument('--batch-size', type=int, default=None,
                    help='training batch size (default from config)')
    pc.add_argument('--train-path', default=None,
                    help='training label.txt / gold TSV path (gives: training phase)')
    pc.add_argument('--val-path', default=None,
                    help='validation label.txt / gold TSV path '
                         '(optional; if omitted, training runs without validation)')
    pc.add_argument('--test-path', default=None,
                    help='test label.txt / gold TSV path (gives: evaluation phase; '
                         'needs --load when given without --train-path)')

    pl = sub.add_parser('lcr', help='train/tune/eval the LCRRothopPP model')
    pl.add_argument('--tune', action='store_true',
                    help='run hyperparameter tuning (requires --train-path)')
    pl.add_argument('--bert-lr', type=float, default=None,
                    help='learning rate for the BERT backbone (default lcr_bert_lr)')
    pl.add_argument('--train-path', default=None,
                    help='training label.txt / gold TSV path (gives: training phase)')
    pl.add_argument('--val-path', default=None,
                    help='validation label.txt / gold TSV path '
                         '(optional; if omitted, training runs without validation)')
    pl.add_argument('--test-path', default=None,
                    help='test label.txt / gold TSV path (gives: evaluation phase; '
                         'needs --load when given without --train-path)')
    pl.add_argument('--epochs', type=int, default=cfg.epochs)
    pl.add_argument('--batch-size', type=int, default=cfg.lcr_batch_size)
    pl.add_argument('--lr', type=float, default=cfg.lcr_learning_rate)
    pl.add_argument('--q', type=float, default=None,
                    help='GCE loss q (default from config)')
    pl.add_argument('--max-length', type=int, default=None,
                    help='max token length for encoding (default from config)')
    pl.add_argument('--hop', type=int, default=cfg.lcr_hop)
    pl.add_argument('--hidden-units', type=int, default=cfg.lcr_hidden_units)
    pl.add_argument('--drop-1', type=float, default=cfg.lcr_drop_1)
    pl.add_argument('--drop-2', type=float, default=cfg.lcr_drop_2)
    pl.add_argument('--l1', type=float, default=cfg.lcr_l1)
    pl.add_argument('--l2', type=float, default=cfg.lcr_l2)
    pl.add_argument('--no-early-stop', action='store_true',
                    help='disable early stopping (train all --epochs regardless of val loss)')
    pl.add_argument('--load', default=None, help='checkpoint name/path to load for eval')
    pl.add_argument('--resume', nargs='?', const=True, default=None,
                    metavar='RUN_DIR|CHECKPOINT',
                    help='resume from checkpoint.pth (of RUN_DIR, a CHECKPOINT file path, '
                         'or the newest run when omitted)')

    return p


HANDLERS = {
    'prep': cmd_prep,
    'casc': cmd_casc,
    'lcr': cmd_lcr,
}


_DEFAULT_RECIPE = {'prep': 'casc', 'casc': 'casc', 'lcr': 'lcr'}
_SUBCOMMANDS = tuple(_DEFAULT_RECIPE)


def main(argv=None):
    # `--config` is global: pre-parse it before build_parser() so the recipe is
    # loaded while argparse defaults (default=training.X) are still read from it.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--config', default=None, metavar='NAME',
                     help='named config recipe (configs/NAME.yaml; '
                          'defaults per subcommand: casc/lcr)')
    pre.add_argument('--device', default=None, help='e.g. cuda:0 or cpu')
    pre_args, rest = pre.parse_known_args(argv)
    # Pin CUDA before lcr_plus_casc.config imports torch (its default device
    # probe runs at import time), so cuda:N maps to cuda:0 inside the process.
    _pin_env(pre_args.device)

    # Drop the consumed --config so the real parser never sees it.
    cleaned, skip = [], False
    for tok in rest:
        if skip:
            skip = False
            continue
        if tok == '--config':
            skip = True
            continue
        if tok.startswith('--config='):
            continue
        cleaned.append(tok)
    rest = cleaned

    command = next((t for t in rest if t in _SUBCOMMANDS), None)
    _load_recipe(pre_args.config or _DEFAULT_RECIPE.get(command, 'casc'))

    args = build_parser().parse_args(rest)
    args._explicit_config = pre_args.config is not None
    _pin_device(args)
    HANDLERS[args.command](args)


if __name__ == '__main__':
    main()
