"""Per-run output directories (output/<config-name>/<timestamp>_<kind>) and run logging."""
import dataclasses
import json
import logging
import os
import sys
from datetime import datetime

from . import config as _config_mod

_NOISY_LOGGERS = ('transformers', 'torch', 'spacy', 'httpx', 'huggingface_hub', 'urllib3')


def artifact_root():
    """output/<config-name> for the active recipe, or None if no recipe is loaded."""
    name = _config_mod.config.get('config_name')
    if not name:
        return None
    return os.path.join(_config_mod.config['output_root'], name)


def start_run(kind, extra=None):
    """Create output/<config-name>/<timestamp>_<kind> and write a config.json snapshot."""
    root = artifact_root()
    if root is None:
        raise RuntimeError('no config recipe loaded; pass --config <name>')
    domain = _config_mod.config['domain']
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(root, f'{stamp}_{kind}')
    os.makedirs(run_dir, exist_ok=True)
    snapshot = {
        'kind': kind,
        'started': datetime.now().isoformat(timespec='seconds'),
        'config_name': _config_mod.config.get('config_name'),
        'config_file': _config_mod.config.get('config_file'),
        'domain': domain,
        'device': _config_mod.config['device'],
        'training': dataclasses.asdict(_config_mod.training),
        'run_dir': run_dir,
    }
    if extra:
        snapshot['args'] = {k: v for k, v in extra.items() if v is not None}
    with open(os.path.join(run_dir, 'config.json'), 'w') as fh:
        json.dump(snapshot, fh, indent=2)
    return run_dir


def latest_artifact(filename, suffixes):
    """Newest <filename> across this recipe's trained run dirs (by name suffix).

    e.g. latest_artifact('casc_model.pth', ('_casc', '_all')). Returns '' if none.
    """
    import glob

    root = artifact_root()
    if root is None or not os.path.isdir(root):
        return ''
    paths = []
    for s in suffixes:
        paths += glob.glob(os.path.join(root, f'*{s}', filename))
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
