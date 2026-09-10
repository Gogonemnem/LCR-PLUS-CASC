import torch as _torch

config = {
    'domain': 'restaurant',
    'device': 'cuda:0' if _torch.cuda.is_available() else 'cpu',
    'output_root': './output',
    'gce_q': 0.4,
}
bert_mapper = {
    'laptop': 'activebus/BERT-DK_laptop',
    'restaurant': 'activebus/BERT-DK_rest'
}
path_mapper = {
    'laptop': './datasets/laptop',
    'restaurant': './datasets/restaurant'
}
aspect_category_mapper = {
    'laptop': ['support', 'os', 'display', 'battery', 'company', 'mouse', 'software', 'keyboard'],
    'restaurant': ['food', 'place', 'service']
}
aspect_seed_mapper = {
    'laptop': {
        'support': {"support", "service", "warranty", "coverage", "replace"},
        'os': {"os", "windows", "ios", "mac", "system", "linux"},
        'display': {"display", "screen", "led", "monitor", "resolution"},
        'battery': {"battery", "life", "charge", "last", "power"},
        'company': {"company", "product", "hp", "toshiba", "dell", "apple", "lenovo"},
        'mouse': {"mouse", "touch", "track", "button", "pad"},
        'software': {"software", "programs", "applications", "itunes", "photo"},
        'keyboard': {"keyboard", "key", "space", "type", "keys"}
    },
    'restaurant': {
        'food': {"food", "spicy", "sushi", "pizza", "taste", "delicious", "bland", "drinks", "flavourful"},
        'place': {"ambience", "atmosphere", "seating", "surroundings", "environment", "location", "decoration", "spacious", "comfortable", "place"},
        'service': {"tips", "manager", "waitress", "rude", "forgetful", "host", "server", "service", "quick", "staff"}
    }
}
sentiment_category_mapper = {
    'laptop': ['negative', 'positive'],
    'restaurant': ['negative', 'positive']
}
sentiment_seed_mapper = {
    'laptop': {
        'positive': {"good", "great", 'nice', "excellent", "perfect", "impressed", "best", "thin", "cheap", "fast"},
        'negative': {"bad", "disappointed", "terrible", "horrible", "small", "slow", "broken", "complaint", "malware", "virus", "junk", "crap", "cramped", "cramp"}
    },
    'restaurant': {
        'positive': {"good", "great", 'nice', "excellent", "perfect", "fresh", "warm", "friendly", "delicious", "fast", "quick", "clean"},
        'negative': {"bad", "terrible", "horrible", "tasteless", "awful", "smelled", "unorganized", "gross", "disappointment", "spoiled", "vomit", "cold", "slow", "dirty", "rotten", "ugly"}
    }
}
M = {
    'laptop': 150,
    'restaurant': 100
}
K_1 = 10
K_2 = 30
lambda_threshold = 0.5
batch_size = 32
validation_data_size = 100
learning_rate = 1e-5
epochs = 20
left_max_length = 100
right_max_length = 100
target_max_length = 10
max_length = 128
spacy_model = 'en_core_web_sm'
test_year = 2016
seed = 0
max_grad_norm = 1.0
test_type = 'test'
label_type = 'single'
# LCR model / training defaults (CLI flags fall back to these)
lcr_learning_rate = 1e-4
lcr_batch_size = batch_size
lcr_hidden_units = 512
lcr_hop = 2
lcr_drop_1 = 0.5
lcr_drop_2 = 0.5
lcr_l1 = 1e-8
lcr_l2 = 1e-8
