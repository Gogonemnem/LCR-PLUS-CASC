import glob
import os

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

from lcr_plus_casc.config import config, domain
from lcr_plus_casc import data

device = config['device']
root_path = domain().root_path

categories = domain().categories
polarities = domain().polarities

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

_, cat_actual, pol_actual = data.load_semeval(year=2016, data_type='test', label_type='single')
print(len(cat_actual))


def _latest_predictions():
    """Newest casc_predictions.csv emitted by the CASC evaluator, under output/ root."""
    root = os.path.abspath(config['output_root'])
    matches = glob.glob(os.path.join(root, '*', '*', 'casc_predictions.csv'))
    if not matches:
        raise FileNotFoundError(
            'no casc_predictions.csv found under '
            f'{root}/<config>/<run>/ -- train first, e.g. `python main.py casc --train-path <train.txt> --test-path <test.txt>`')
    return max(matches, key=os.path.getmtime)


preds_path = _latest_predictions()
print(f'Loaded predictions from {preds_path}')
df = pd.read_csv(preds_path)
cat_pred = [inv_aspect_dict[c] for c in df['predicted category']]
pol_pred = [inv_polarity_dict[p] for p in df['predicted polarity']]

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




