"""Single entry point for the CASC (BERT) and LCR (RoTHop++) pipelines.

Subcommands
-----------
  prep     Build vocabularies / extract spans / score / generate labels (CASC preprocessing).
  embed    Produce combined BERT-DK embeddings for a split (used by LCR).
  casc     Train and/or evaluate the BERTLinear CASC model.
  lcr      Train / tune / evaluate the two-tower LCRRothopPP model.
  all      Run everything: casc (train+eval) then lcr (train+eval) in one shot.

Each training run writes artifacts to output/<domain>/<timestamp>_<kind>/:
  config.json, run.log, predictions.csv, plus per-model artifacts:
  casc_model.pth / casc_checkpoint.pth (CASC), lcr_model.pth / lcr_checkpoint.pth (LCR).

Resume a crashed run (after N more epochs are added automatically) with:
  python main.py casc --resume output/restaurant/<run-dir>

Examples
--------
  # everything at once
  python main.py --domain restaurant --device cuda:0 all

  # just the LCR model on a held-out test split
  python main.py lcr --run both

  # just the CASC model
  python main.py casc --run train --epochs 5

  # resume from the latest checkpoint of a previous run
  python main.py casc --run train --epochs 20 --resume

  # hyperparameter search
  python main.py lcr --run tune
"""
import argparse
import logging
import os


def _config(args):
    """Apply --domain/--device/--gce-q to the package config before importing data modules."""
    import lcr_plus_casc.config as cfg
    if args.domain:
        cfg.config['domain'] = args.domain
    if args.device:
        cfg.config['device'] = args.device
    if getattr(args, 'gce_q', None) is not None:
        cfg.config['gce_q'] = args.gce_q
    if getattr(args, 'seed', None) is not None:
        cfg.seed = args.seed
    return cfg


def _resume_checkpoint_names(kind):
    if kind == 'casc':
        return ['casc_checkpoint.pth']
    if kind == 'lcr':
        return ['lcr_checkpoint.pth']
    return ['casc_checkpoint.pth', 'lcr_checkpoint.pth']  # shared `all` run dir


def _resolve_resume(resume, kind):
    """--resume <run-dir> uses that dir; bare --resume reuses the newest existing run."""
    import glob
    import lcr_plus_casc.config as cfg
    if not resume:
        return None
    names = _resume_checkpoint_names(kind)
    if isinstance(resume, str):
        if any(os.path.isfile(os.path.join(resume, n)) for n in names):
            return resume
    # Dedicated <kind> runs plus shared `all` runs (checkpoints land in both)
    suffixes = [f'_{kind}', '_all'] if kind in ('casc', 'lcr') else [f'_{kind}']
    matches = []
    for suffix in suffixes:
        for n in names:
            matches.extend(glob.glob(os.path.join(cfg.config['output_root'], cfg.config['domain'], f'*{suffix}', n)))
    if matches:
        old = os.path.dirname(sorted(matches)[-1])
        logging.getLogger(__name__).info('resuming run %s', old)
        return old
    raise SystemExit(f'No runnable checkpoint ({", ".join(names)}) found to resume ({kind})')


# ------------------------------- preprocessing --------------------------------
def cmd_prep(args):
    _config(args)
    from lcr_plus_casc.labeling import run_labeling

    run_labeling()
    print('prep complete')


# --------------------------------- embeddings ---------------------------------
def cmd_embed(args):
    import numpy as np
    _config(args)
    from lcr_plus_casc import data
    from lcr_plus_casc.classifiers import tokenize_separated, embed_separated
    import os

    sentences, cats, pols = data.load_training_data(training_path=args.path)
    os.makedirs(args.output, exist_ok=True)

    embs = []
    for i, s in enumerate(sentences):
        if i and i % 50 == 0:
            print(f'embedded {i}/{len(sentences)}')
        tokens = tokenize_separated(s)
        emb = embed_separated(tokens)              # [1, total_len, 768] combined tensor
        embs.append(emb[0])
    all_embs = np.asarray(embs, dtype='float32')   # [N, total_len, 768]
    for i, e in enumerate(all_embs):
        np.save(f'{args.output}/{i}.npy', e.reshape(1, *e.shape))
    if not args.output.endswith('s.npy') and not args.output.endswith('.npy'):
        args.output = f'{args.output}s'
    np.save(f'{args.output}.npy', all_embs)
    print(f'saved {all_embs.shape} to {args.output}.npy')


def _ensure_run(args, kind, extra=None):
    """Pick the run dir: --resume dir > dir already started this invocation > new.

    Sets config['run_dir'] and wires the run.log handlers; returns the run dir.
    """
    import lcr_plus_casc.config as cfg
    from lcr_plus_casc import runs

    resume_dir = (_resolve_resume(args.resume, kind)
                  if getattr(args, 'resume', None) else None)
    if resume_dir:
        run_dir = resume_dir
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
    """Best model path for eval: this run dir first, else the newest trained run."""
    from lcr_plus_casc import runs

    local = os.path.join(run_dir, f'{filename}.pth')
    if os.path.isfile(local):
        return local
    path = runs.latest_artifact(f'{filename}.pth', suffixes)
    if not path and filename != 'best_model':
        # Legacy runs stored best_model.pth; in shared *_all dirs it holds the
        # LCR weights (trained last), so only trust it for LCR suffixed runs.
        legacy = ('_casc',) if 'casc' in filename else suffixes
        path = runs.latest_artifact('best_model.pth', legacy)
    return path or filename


# ----------------------------------- CASC -------------------------------------
def cmd_casc(args):
    _config(args)
    from lcr_plus_casc.classifiers.training import CASC

    kind = 'all' if args.command == 'all' else 'casc'
    run_dir = _ensure_run(args, kind, extra={'epochs': args.epochs, 'device': args.device})

    trainer = CASC()
    resume_ckpt = os.path.join(run_dir, 'casc_checkpoint.pth') if args.resume else None
    if args.run in ('train', 'both'):
        dataset = trainer.load_training_data()
        trainer.train_model(dataset, epochs=args.epochs, resume=resume_ckpt)
        trainer.save_model(args.save or 'casc_model')
    if args.run in ('eval', 'both'):
        trainer.load_model(args.load or _resolve_load(run_dir, 'casc_model', ('_casc', '_all')))
        trainer.evaluate(year=args.test_year, test_type=args.test_type,
                         label_type=args.label_type)


# ----------------------------------- LCR --------------------------------------
def _load_lcr_data(args):
    from lcr_plus_casc import data
    X_train, cat_train, pol_train = data.load_embedded_training()
    X_val = cat_val = pol_val = None
    if args.val_emb:
        X_val, cat_val, pol_val = data.load_embedded_split(args.val_emb, args.val_path)
    return X_train, cat_train, pol_train, X_val, cat_val, pol_val


def cmd_lcr(args):
    import numpy as np
    cfg = _config(args)
    from lcr_plus_casc.classifiers.training import LCR

    X_train, cat_train, pol_train, X_val, cat_val, pol_val = _load_lcr_data(args)
    domain = cfg.config['domain']

    def make_trainer():
        return LCR(hidden_units=args.hidden_units, hop=args.hop,
                   drop_1=args.drop_1, drop_2=args.drop_2,
                   num_cat=len(cfg.aspect_category_mapper[domain]),
                   num_pol=len(cfg.sentiment_category_mapper[domain]))

    if args.run == 'tune':
        _ensure_run(args, 'lcr_tune', extra={'device': args.device})
        trainer = make_trainer()
        if X_val is None:
            import torch
            ds = torch.utils.data.TensorDataset(
                torch.from_numpy(np.asarray(X_train)),
                torch.as_tensor(cat_train), torch.as_tensor(pol_train))
            _, vl = torch.utils.data.random_split(ds, [len(ds) - 100, 100])
            X_val = np.stack([d[0].numpy() for d in vl])
            cat_val = [d[1].item() for d in vl]
            pol_val = [d[2].item() for d in vl]
        trainer.tune(X_train, cat_train, pol_train, X_val, cat_val, pol_val,
                     max_epochs=args.epochs)
        return

    run_dir = _ensure_run(args, 'lcr', extra={'epochs': args.epochs, 'device': args.device})

    if args.run in ('train', 'both'):
        trainer = make_trainer()
        trainer.train_model(X_train, cat_train, pol_train, X_val, cat_val, pol_val,
                            epochs=args.epochs, batch_size=args.batch_size,
                            learning_rate=args.lr, q=args.q, l1=args.l1, l2=args.l2,
                            resume=os.path.join(run_dir, 'lcr_checkpoint.pth')
                            if args.resume else None)

    if args.run in ('eval', 'both'):
        trainer = make_trainer()
        load_arg = args.load
        if load_arg and not (os.path.isdir(load_arg) or load_arg.endswith('.pth')):
            load_arg = _resolve_load(run_dir, 'lcr_model', ('_lcr', '_all'))
        trainer.load_model(load_arg or _resolve_load(run_dir, 'lcr_model', ('_lcr', '_all')))
        if args.test_emb:
            from lcr_plus_casc import data
            X_test, cat_test, pol_test = data.load_embedded_split(args.test_emb, args.test_path)
            trainer.evaluate(X_test, cat_test, pol_test)


# ------------------------------------ all -------------------------------------
def cmd_all(args):
    cmd_casc(args)
    cmd_lcr(args)


# --------------------------------- CLI setup ----------------------------------
def build_parser():
    from lcr_plus_casc import config as cfg
    p = argparse.ArgumentParser(description='CASC + LCR aspect/polarity pipeline')
    p.add_argument('--domain', default=None, choices=['restaurant', 'laptop'])
    p.add_argument('--device', default=None, help="e.g. cuda:0 or cpu")
    p.add_argument('--seed', type=int, default=None,
                   help='random seed (default from config)')
    sub = p.add_subparsers(dest='command', required=True)

    sub.add_parser('prep', help='CASC preprocessing (vocab/extract/score/label)')

    pe = sub.add_parser('embed', help='produce combined BERT-DK embeddings for a split')
    pe.add_argument('--path', default=None, help='split label.txt path')
    pe.add_argument('--output', default=None, help='embedding output folder')

    pc = sub.add_parser('casc', help='train/eval the BERTLinear CASC model')
    pc.add_argument('--run', default='both', choices=['train', 'eval', 'both'])
    pc.add_argument('--load', default=None, help='checkpoint name to load for eval')
    pc.add_argument('--save', default=None, help='checkpoint name to save after train')
    pc.add_argument('--resume', nargs='?', const=True, default=None,
                    metavar='RUN_DIR',
                    help='resume from checkpoint.pth (of RUN_DIR, or the newest run when omitted)')
    pc.add_argument('--epochs', type=int, default=cfg.epochs)
    pc.add_argument('--gce-q', type=float, default=None,
                    help='GCE loss q (default from config: 0.4)')
    pc.add_argument('--test-year', type=int, default=None,
                    help='SemEval test year (default from config)')
    pc.add_argument('--test-type', default=cfg.test_type)
    pc.add_argument('--label-type', default=cfg.label_type)

    pl = sub.add_parser('lcr', help='train/tune/eval the LCRRothopPP model')
    pl.add_argument('--run', default='both', choices=['train', 'eval', 'tune', 'both'])
    pl.add_argument('--val-emb', default=None, help='validation embedding folder')
    pl.add_argument('--val-path', default=None, help='validation label.txt path')
    pl.add_argument('--test-emb', default=None, help='test embedding folder')
    pl.add_argument('--test-path', default=None, help='test label.txt path')
    pl.add_argument('--epochs', type=int, default=cfg.epochs)
    pl.add_argument('--batch-size', type=int, default=cfg.lcr_batch_size)
    pl.add_argument('--lr', type=float, default=cfg.lcr_learning_rate)
    pl.add_argument('--q', type=float, default=None,
                    help='GCE loss q (default from config)')
    pl.add_argument('--hop', type=int, default=cfg.lcr_hop)
    pl.add_argument('--hidden-units', type=int, default=cfg.lcr_hidden_units)
    pl.add_argument('--drop-1', type=float, default=cfg.lcr_drop_1)
    pl.add_argument('--drop-2', type=float, default=cfg.lcr_drop_2)
    pl.add_argument('--l1', type=float, default=cfg.lcr_l1)
    pl.add_argument('--l2', type=float, default=cfg.lcr_l2)
    pl.add_argument('--load', default=None, help='checkpoint name/path to load for eval')
    pl.add_argument('--resume', nargs='?', const=True, default=None,
                    metavar='RUN_DIR',
                    help='resume from checkpoint.pth (of RUN_DIR, or the newest run when omitted)')

    pa = sub.add_parser('all', help='run casc then lcr (train+eval) in one shot')
    pa.add_argument('--run', default='both', choices=['train', 'eval', 'both'],
                    help='applied to both casc and lcr')
    pa.add_argument('--epochs', type=int, default=cfg.epochs,
                    help='applied to both casc and lcr')
    pa.add_argument('--resume', nargs='?', const=True, default=None, metavar='RUN_DIR',
                    help='resume both models from the given run dir (newest run when omitted)')
    return p


HANDLERS = {
    'prep': cmd_prep,
    'embed': cmd_embed,
    'casc': cmd_casc,
    'lcr': cmd_lcr,
    'all': cmd_all,
}


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == 'all':
        # The bare `all` subparser has no per-model flags; set shared defaults.
        import argparse
        import lcr_plus_casc.config as cfgc
        for attr, val in [('run', 'both'), ('epochs', cfgc.epochs),
                          ('batch_size', cfgc.lcr_batch_size), ('lr', cfgc.lcr_learning_rate),
                          ('q', None), ('hop', cfgc.lcr_hop),
                          ('hidden_units', cfgc.lcr_hidden_units),
                          ('drop_1', cfgc.lcr_drop_1), ('drop_2', cfgc.lcr_drop_2),
                          ('l1', cfgc.lcr_l1), ('l2', cfgc.lcr_l2),
                          ('save', None), ('load', None), ('resume', None),
                          ('test_year', None), ('test_type', cfgc.test_type),
                          ('label_type', cfgc.label_type),
                          ('val_emb', None), ('val_path', None),
                          ('test_emb', None), ('test_path', None)]:
            if not hasattr(args, attr):
                setattr(args, attr, val)
    if args.command == 'embed':
        if not args.path:
            import lcr_plus_casc.config as cfg
            args.path = f"{cfg.path_mapper[cfg.config['domain']]}/label.txt"
        if not args.output:
            import lcr_plus_casc.config as cfg
            args.output = f"{cfg.path_mapper[cfg.config['domain']]}/training_embedding"
    HANDLERS[args.command](args)


if __name__ == '__main__':
    main()
