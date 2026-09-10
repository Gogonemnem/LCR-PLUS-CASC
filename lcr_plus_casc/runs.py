"""Per-run output directories (output/<domain>/<timestamp>) and run logging."""
import json
import logging
import os
import sys
from datetime import datetime

from . import config as _config_mod

_NOISY_LOGGERS = ('transformers', 'torch', 'spacy', 'httpx', 'huggingface_hub', 'urllib3')


def start_run(kind, extra=None):
    """Create output/<domain>/<timestamp>_<kind> and write a config.json snapshot."""
    domain = _config_mod.config['domain']
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(_config_mod.config['output_root'], domain, f'{stamp}_{kind}')
    os.makedirs(run_dir, exist_ok=True)
    snapshot = {
        'kind': kind,
        'started': datetime.now().isoformat(timespec='seconds'),
        'domain': domain,
        'device': _config_mod.config['device'],
        'learning_rate': _config_mod.learning_rate,
        'epochs': _config_mod.epochs,
        'batch_size': _config_mod.batch_size,
        'validation_data_size': _config_mod.validation_data_size,
        'run_dir': run_dir,
    }
    if extra:
        snapshot['args'] = {k: v for k, v in extra.items() if v is not None}
    with open(os.path.join(run_dir, 'config.json'), 'w') as fh:
        json.dump(snapshot, fh, indent=2)
    return run_dir


def latest_artifact(filename, suffixes):
    """Newest <filename> across this domain's trained run dirs (by name suffix).

    e.g. latest_artifact('best_model.pth', ('_lcr', '_all')). Returns '' if none.
    """
    import glob

    domain = _config_mod.config['domain']
    root = _config_mod.config['output_root']
    paths = []
    for s in suffixes:
        paths += glob.glob(os.path.join(root, domain, f'*{s}', filename))
    return max(paths) if paths else ''


def setup_logging(run_dir, level=logging.INFO):
    """Log to <run_dir>/run.log plus the console."""
    fmt = logging.Formatter('%(asctime)s %(levelname)s %(name)s - %(message)s')

    root = logging.getLogger()
    root.setLevel(level)
    for h in list(root.handlers):
        if isinstance(h, logging.FileHandler) or h.name == 'run-console':
            root.removeHandler(h)

    fh = logging.FileHandler(os.path.join(run_dir, 'run.log'))
    fh.setFormatter(fmt)
    root.addHandler(fh)

    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(fmt)
    sh.name = 'run-console'
    root.addHandler(sh)

    for noisy in _NOISY_LOGGERS:
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return logging.getLogger('lcr_plus_casc')
