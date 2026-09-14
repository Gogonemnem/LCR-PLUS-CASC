import os

from .config import config, domain as _domain
from .classifiers.lcr import encode_separated_batch
from .labeling.split_file import read_labeled

import torch

_dcfg = _domain()
root_path = _dcfg.root_path

categories = _dcfg.categories
polarities = _dcfg.polarities
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

def _to_int(label, inv_dict):
    """Map a category/polarity label to its integer index.

    Accepts a string name (e.g. 'food'/'positive') as produced by the CASC
    labeling stage, or an integer (SemEval gold files store 0/1/2/3/4/5).
    """
    if isinstance(label, str) and label in inv_dict:
        return inv_dict[label]
    return int(label)


# Raw SemEval gold polarity encoding (see labeling/semeval_reader.POLARITY_MAP):
# the 3-class 2016 gold stores negative/-1, neutral/0, positive/1.
_SEMEVAL_POL_TO_NAME = {1: 'positive', 0: 'neutral', -1: 'negative'}


def _semeval_pol_index(pol):
    """Map a raw SemEval gold polarity (-1/0/1) to the model's polarity index.

    The SemEval gold is 3-class (negative/neutral/positive) but the ABSA model
    is 2-class (negative/positive). Polarity names the model recognises map
    directly; any class the model lacks (neutral) folds into the non-positive
    bucket so the evaluation is well-formed for the model's head.
    """
    name = _SEMEVAL_POL_TO_NAME.get(int(pol))
    if name is not None and name in inv_polarity_dict:
        return inv_polarity_dict[name]
    return inv_polarity_dict.get('negative', 0)


def _pol_from_cell(cell):
    """Map a training-file polarity cell to the model's 2-class index.

    Domain names ('negative'/'positive') map straight to their index; raw
    SemEval gold codes (-1/0/1, the 3-class gold) fold to 2-class via
    _semeval_pol_index so training labels match the evaluation folds.
    """
    if isinstance(cell, str) and cell.strip() in inv_polarity_dict:
        return inv_polarity_dict[cell.strip()]
    return _semeval_pol_index(cell)


def load_training_data(training_path=None, **kwargs):
    """Read a labeled training set (label.txt rows or SemEval-style gold TSV).

    Uses labeling.read_labeled, which auto-detects TSV (idx/cat/pol/sentence)
    vs the legacy alternating 'sentence\n label\n' layout, so both formats load
    identically. Sentences keep their raw form: labeler rows carry ``# # #``
    target separators (or are plain), gold rows are plain sentences.
    """
    if training_path is None:
        training_path = f'{root_path}/label.txt'
    rows = read_labeled(training_path)
    sentences = [row[3] for row in rows]
    cats = [_to_int(row[1], inv_aspect_dict) for row in rows]
    pols = [_pol_from_cell(row[2]) for row in rows]
    return sentences, cats, pols


def make_tensor_dataset(sentences, cats, pols):
    """Tokenize ' [SEP] '-separated sentences into the in-graph LCR dataset.

    Returns a TensorDataset of (input_ids [N, L], attention_mask [N, L],
    ranges [N, 6] long, cats long, pols long).
    """
    from torch.utils.data import TensorDataset
    input_ids, attention_mask, ranges = encode_separated_batch(sentences)
    return TensorDataset(
        input_ids,
        attention_mask,
        ranges,
        torch.as_tensor(cats, dtype=torch.long),
        torch.as_tensor(pols, dtype=torch.long),
    )


def load_semeval(year, data_type, label_type, **kwargs):
    path = f'{root_path}/{str(year)}/{data_type}_{label_type}.txt'
    if not os.path.exists(path):
        path = f'{root_path}/raw/{data_type}.txt'
    with open(path, 'r', encoding='utf-8') as f:
        sentences = []
        cats = []
        pols = []

        for line in f:
            line = line.rstrip('\n')
            if not line.strip():
                continue
            # maxsplit keeps any embedded tabs inside the sentence intact.
            # The ' [SEP] ' markers in sentences must be preserved verbatim.
            split_line = line.split('\t', 3)
            if len(split_line) < 4:
                continue

            _, cat, pol, sentence = split_line
            cats.append(int(cat))
            pols.append(_semeval_pol_index(pol))
            sentences.append(sentence)

        return sentences, cats, pols

def main():
    ss, cs, ps = load_training_data()
    print(len(ss), len(cs), len(ps))
    print(cs.count(0), cs.count(1), cs.count(2))
    print(ps.count(0), ps.count(1))


if __name__ == '__main__':
    main()
    
