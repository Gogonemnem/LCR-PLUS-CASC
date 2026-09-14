"""Package configuration: a TrainingConfig singleton plus per-domain DomainConfig instances.

The `config` dict holds only runtime-mutable state (active domain, device, output
root, GCE loss q, metric list, tokenization length, and the active run dir set
by main). Everything else lives in `training` (TrainingConfig) or the
per-domain DOMAINS entries (DomainConfig), accessed via `domain()`.

Named recipes live in the repo-root `configs/` directory as YAML files: a
partial override of `TrainingConfig` fields plus optional `domain`, `device`,
`output_root` and the recipe's own `name`. Load with `load_config_file()`
(populates the singletons) or `apply_config()` (dict, e.g. a CLI parse result).
"""
from dataclasses import asdict, dataclass, fields
import os

import torch as _torch


@dataclass
class TrainingConfig:
    """Default training / preprocessing hyperparameters (overridable via CLI)."""
    batch_size: int = 32
    learning_rate: float = 1e-5
    lcr_learning_rate: float = 1e-4
    lcr_bert_lr: float = 1e-5
    lcr_batch_size: int = 32
    lcr_hidden_units: int = 512
    lcr_hop: int = 2
    lcr_drop_1: float = 0.5
    lcr_drop_2: float = 0.5
    lcr_l1: float = 1e-8
    lcr_l2: float = 1e-8
    epochs: int = 20
    max_grad_norm: float = 1.0
    seed: int = 0
    validation_data_size: int = 100
    test_year: int = 2016
    test_type: str = 'test'
    label_type: str = 'single'
    left_max_length: int = 100
    right_max_length: int = 100
    target_max_length: int = 10
    gce_q: float = 0.4
    lcr_embed_type: str = 'combined'
    log_every: int = 200

    # --- weak-supervision / prep labeling knobs (divergence alignment paper<->reference) ---
    # div 1: how per-category candidates combine. 'max' = per-token max over top-K replacements
    #        (paper "maximum score labeler", ours); 'sum' = sum of matching replacements over all
    #        aspect-word occurrences divided by the occurrence count (reference avg).
    label_agg: str = 'max'
    # div 2: score assigned to a category with no matching candidate ('-1' ours; 0.0 reference).
    label_score_default: float = -1.0
    # div 3: z-score population. 'filtered' = over POS-filtered rows Xq only (paper, ours);
    #        'all' = over every sentence, ## rows scored 0.0 (reference -> smaller sigma).
    label_z_population: str = 'filtered'
    # div 4: require the thresholded aspect word to be regex-located before labeling.
    #        True = drop rows where it is not found (paper/ours, downstream-safe);
    #        False = keep the row, writing the plain sentence when it cannot be located (reference).
    label_require_word: bool = True

    @property
    def max_length(self):
        """Total encoded length: [CLS] left [SEP] target [SEP] right [SEP]."""
        return self.left_max_length + self.target_max_length + self.right_max_length + 4


@dataclass
class DomainConfig:
    """Per-dataset (laptop / restaurant) model, data and labeling settings."""
    name: str
    bert_model: str
    root_path: str
    categories: tuple
    polarities: tuple
    aspect_seeds: dict
    sentiment_seeds: dict
    K_1: int = 20
    K_2: int = 20
    M: int = 100
    lambda_threshold: float = 0.5
    spacy_model: str = 'en_core_web_sm'


LAPTOP = DomainConfig(
    name='laptop',
    bert_model='activebus/BERT-DK_laptop',
    root_path='./datasets/laptop',
    categories=('support', 'os', 'display', 'battery', 'company', 'mouse', 'software', 'keyboard'),
    polarities=('negative', 'positive'),
    aspect_seeds={
        'support': {"support", "service", "warranty", "coverage", "replace"},
        'os': {"os", "windows", "ios", "mac", "system", "linux"},
        'display': {"display", "screen", "led", "monitor", "resolution"},
        'battery': {"battery", "life", "charge", "last", "power"},
        'company': {"company", "product", "hp", "toshiba", "dell", "apple", "lenovo"},
        'mouse': {"mouse", "touch", "track", "button", "pad"},
        'software': {"software", "programs", "applications", "itunes", "photo"},
        'keyboard': {"keyboard", "key", "space", "type", "keys"},
    },
    sentiment_seeds={
        'positive': {"good", "great", 'nice', "excellent", "perfect", "impressed", "best", "thin", "cheap", "fast"},
        'negative': {"bad", "disappointed", "terrible", "horrible", "small", "slow", "broken", "complaint", "malware", "virus", "junk", "crap", "cramped", "cramp"},
    },
    M=150,
)

RESTAURANT = DomainConfig(
    name='restaurant',
    bert_model='activebus/BERT-DK_rest',
    root_path='./datasets/restaurant',
    categories=('food', 'place', 'service'),
    polarities=('negative', 'positive'),
    aspect_seeds={
        'food': {"food", "spicy", "sushi", "pizza", "taste", "delicious", "bland", "drinks", "flavourful"},
        'place': {"ambience", "atmosphere", "seating", "surroundings", "environment", "location", "decoration", "spacious", "comfortable", "place"},
        'service': {"tips", "manager", "waitress", "rude", "forgetful", "host", "server", "service", "quick", "staff"},
    },
    sentiment_seeds={
        'positive': {"good", "great", 'nice', "excellent", "perfect", "fresh", "warm", "friendly", "delicious", "fast", "quick", "clean"},
        'negative': {"bad", "terrible", "horrible", "tasteless", "awful", "smelled", "unorganized", "gross", "disappointment", "spoiled", "vomit", "cold", "slow", "dirty", "rotten", "ugly"},
    },
)

DOMAINS = {d.name: d for d in (LAPTOP, RESTAURANT)}

# Mutable singleton: tests / CLI mutate fields (e.g. training.validation_data_size).
training = TrainingConfig()

# Repo-root `configs/` directory: one YAML recipe per named experiment.
CONFIGS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'configs')

# Canonical checkpoint file names per model kind.
CKPT = {'casc': 'casc_checkpoint.pth', 'lcr': 'lcr_checkpoint.pth'}

# Runtime-mutable state only; set/updated by main.py before importing data modules.
config = {
    'domain': 'restaurant',
    'device': 'cuda' if _torch.cuda.is_available() else 'cpu',
    'output_root': './output',
    'gce_q': training.gce_q,
    # metrics reported by both models via the shared evaluate (keys become cat_<name> / pol_<name>)
    'metrics': ['acc', 'f1'],
    'max_length': training.max_length,
    # selected recipe (None when running on plain defaults); set by load_config_file()
    'config_name': None,
    'config_file': None,
}

# Top-level keys of a recipe YAML. `name` is optional (defaults to the file stem);
# `domain` selects the DomainConfig; `extra` (arbitrary) is stored for the snapshot.
_TRAINING_FIELDS = {f.name for f in fields(TrainingConfig)}
_KNOWN_TOP_KEYS = {'name', 'domain', 'device', 'output_root', 'extra'} | _TRAINING_FIELDS


def domain(name=None):
    """DomainConfig for the active domain (config['domain']), or `name` when given."""
    return DOMAINS[name or config['domain']]


def _config_error(msg, file_path=None):
    where = f' in {file_path}' if file_path else ''
    raise ValueError(f'config error{where}: {msg}')


def _apply_overrides(data, file_path=None):
    """Apply a recipe mapping onto the singletons; return a summary dict."""
    if not isinstance(data, dict):
        _config_error('top level must be a mapping', file_path)

    data = dict(data)
    explicit_name = data.pop('name', None)
    extra = data.pop('extra', None)
    device = data.pop('device', None)
    domain_name = data.pop('domain', None)
    output_root = data.pop('output_root', None)

    unknown = sorted(set(data) - _TRAINING_FIELDS)
    if unknown:
        allowed = sorted(_TRAINING_FIELDS | _KNOWN_TOP_KEYS)
        _config_error(f"unknown keys: {unknown} (allowed: {allowed})", file_path)

    for key, value in data.items():
        if not isinstance(key, str):
            _config_error(f"non-string training field {key!r}", file_path)
        if value is None:
            continue
        old = getattr(training, key)
        if old is not None and not isinstance(value, type(old)) and \
                not (isinstance(old, bool) and isinstance(value, bool)):
            _config_error(f"field '{key}' expects {type(old).__name__}, got {type(value).__name__}: {value!r}", file_path)
        setattr(training, key, value)

    if domain_name is not None:
        try:
            config['domain'] = domain(domain_name).name
        except KeyError:
            _config_error(f"unknown domain {domain_name!r} (known: {sorted(DOMAINS)})", file_path)
    if device is not None:
        config['device'] = device
    if output_root is not None:
        config['output_root'] = output_root
    config['gce_q'] = training.gce_q
    config['max_length'] = training.max_length

    name = explicit_name
    if name is None and file_path:
        import os as _os
        name = _os.path.splitext(_os.path.basename(file_path))[0]
    if name is not None:
        config['config_name'] = name
    if file_path is not None:
        config['config_file'] = file_path

    return {
        'name': config['config_name'],
        'file': config['config_file'],
        'domain': config['domain'],
        'training': asdict(training),
        'extra': extra,
    }


def resolve_config(path_or_name):
    """Resolve a `--config` value to an absolute recipe file path (or None).

    Accepts an existing path (e.g. `configs/lcr_l2.yaml`, `./recipes/x.yml`)
    or a bare name resolved against the repo-root `configs/` directory.
    """
    import os
    p = str(path_or_name)
    if os.path.exists(p):
        return os.path.abspath(p)
    for ext in ('.yaml', '.yml'):
        candidate = os.path.join(CONFIGS_DIR, p + ext)
        if os.path.isfile(candidate):
            return candidate
    for candidate in (os.path.join(CONFIGS_DIR, p), os.path.abspath(p)):
        if os.path.isfile(candidate):
            return candidate
    available = sorted(f for f in os.listdir(CONFIGS_DIR)
                       if f.endswith(('.yaml', '.yml'))) if os.path.isdir(CONFIGS_DIR) else []
    _config_error(f"no config file '{p}' found (configs root: {CONFIGS_DIR}; available: {available})")
    return None


def load_config_file(path_or_name):
    """Load a YAML recipe and apply it to the singletons; return its summary.

    A recipe is a partial `TrainingConfig`: only listed fields are overridden,
    missing ones keep dataclass defaults / prior CLI values. Optional top-level
    `name`, `domain`, `device`, `output_root` and `extra` are also supported.
    """
    import os

    file_path = resolve_config(path_or_name)
    if file_path is None:
        return None
    try:
        import yaml
    except ImportError:
        raise RuntimeError("reading YAML configs requires pyyaml (pip install pyyaml)")
    with open(file_path) as fh:
        try:
            data = yaml.safe_load(fh) or {}
        except yaml.YAMLError as exc:
            _config_error(f'invalid YAML: {exc}', file_path)
    return _apply_overrides(data, file_path)


def apply_config(overrides):
    """Apply a plain mapping of TrainingConfig fields (e.g. CLI/argparse result)."""
    return _apply_overrides(dict(overrides or {}))
