from config import *
import data

domain = config['domain']
device = config['device']
root_path = path_mapper[domain]

categories = aspect_category_mapper[domain]
polarities = sentiment_category_mapper[domain]

aspect_dict = {}
inv_aspect_dict = {}
for i, cat in enumerate(categories):
    aspect_dict[i] = cat
    inv_aspect_dict[cat] = i

polarity_dict = {}
inv_polarity_dict = {}
for i, pol in enumerate(polarities):
    polarity_dict[i] = pol
    inv_polarity_dict[pol] = i

_, cat_actual, pol_actual = data.load_embedded(data.load_semeval, year=2016, data_type='test', label_type='single')
print(len(cat_actual))
with open(r'datasets\restaurant\label 2016 single.txt') as f:
    sentences = []
    cat_pred = []
    pol_pred = []
    for idx, line in enumerate(f):
        if idx % 2 == 1:
            cat, pol = line.replace(" [SEP] ", "").strip().split()
            cat_pred.append(inv_aspect_dict[cat])
            pol_pred.append(inv_polarity_dict[pol])
        else:
            sentences.append(line.strip())

import numpy as np
from sklearn.metrics import classification_report

predicted = np.array(pol_pred)
actual = np.array(pol_actual)
print("Polarity")
print(classification_report(actual, predicted, digits=4))
print()

predicted = np.array(cat_pred)
actual = np.array(cat_actual)
print("Aspect")
print(classification_report(actual, predicted, digits=4))
print()




