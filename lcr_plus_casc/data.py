import os

from .config import (
    aspect_category_mapper,
    config,
    path_mapper,
    sentiment_category_mapper,
)
from .classifiers.embedding import embed_separated, tokenize_separated
from .labeling.split_file import read_labeled

import numpy as np
import torch

domain = config['domain']
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

def _to_int(label, inv_dict):
    """Map a category/polarity label to its integer index.

    Accepts a string name (e.g. 'food'/'positive') as produced by the CASC
    labeling stage, or an integer (SemEval gold files store 0/1/2/3/4/5).
    """
    if isinstance(label, str) and label in inv_dict:
        return inv_dict[label]
    return int(label)


def load_training_data(training_path=None, **kwargs):
    """Read the CASC-labeled training set.

    Uses labeling.read_labeled, which auto-detects TSV (idx/cat/pol/sentence)
    vs the legacy alternating 'sentence\n label\n' layout, so both formats load
    identically.
    """
    if training_path is None:
        training_path = f'{root_path}/label.txt'
    rows = read_labeled(training_path)
    sentences = [row[3] for row in rows]
    cats = [_to_int(row[1], inv_aspect_dict) for row in rows]
    pols = [_to_int(row[2], inv_polarity_dict) for row in rows]
    return sentences, cats, pols


def load_embedded_split(embed_folder, label_path):
    """Load an embedded split: combined embedding tensor + int category/polarity labels.

    `embed_folder` may omit the trailing 's'; the combined file is '<folder>s.npy'.
    """
    ss, cs, ps = load_training_data(training_path=label_path)
    if not embed_folder.endswith('s.npy') and not embed_folder.endswith('.npy'):
        embed_folder = f'{embed_folder}s'
    emb = np.load(f'{embed_folder}.npy')
    if emb.shape[0] != len(cs):
        raise ValueError(f'Embedding/label shape mismatch in {embed_folder}: {emb.shape[0]} vs {len(cs)}')
    return emb, cs, ps


def make_tensor_dataset(X, cats, pols):
    """Build a torch TensorDataset from a combined embedding array + int labels."""
    from torch.utils.data import TensorDataset
    return TensorDataset(
        torch.from_numpy(np.ascontiguousarray(np.asarray(X, dtype=np.float32))),
        torch.as_tensor(cats, dtype=torch.long),
        torch.as_tensor(pols, dtype=torch.long),
    )


def save_in_separate(sentences, folder_path):
    os.makedirs(folder_path, exist_ok=True)
    total = len(str(len(sentences)))

    for i, sent in enumerate(sentences):
        tokens = tokenize_separated(sent)
        embedding = embed_separated(tokens)
        number = str(i).rjust(total, '0')
        np.save(f'{folder_path}/{number}', embedding)

def save_to_single(folder_path):
    
    files = os.listdir(folder_path)

    first = np.load(f'{folder_path}/{files[0]}')
    all_embeddings = np.zeros((len(files), first.shape[1], first.shape[2]))

    for i, file in enumerate(files):
        array = np.load(f'{folder_path}/{file}')
        all_embeddings[i, :, :] = array[0, :, :]

    np.save(f'{folder_path}s', all_embeddings)

def load_embedded_training(**kwargs):
    ss, cs, ps = load_training_data()
    emb = np.load(f'{root_path}/training_embeddings.npy')
    return emb, cs, ps

def load_semeval(year, data_type, label_type, **kwargs):
    path = f'{root_path}/{str(year)}/{data_type}_{label_type}.txt'
    if not os.path.exists(path):
        path = f'{root_path}/{data_type}.txt'
    with open(path, 'r', encoding='utf-8') as f:
        sentences = []
        cats = []
        pols = []

        for line in f:
            stripped = line.strip().replace(' [SEP] ', '')
            # maxsplit keeps any embedded tabs inside the sentence intact.
            split_line = stripped.split('\t', 3)
            if len(split_line) < 4:
                continue

            _, cat, pol, sentence = split_line
            cats.append(int(cat))
            pols.append(int(pol))
            sentences.append(sentence)

        return sentences, cats, pols

def load_embedded(load_func, **kwargs):
    ss, cs, ps = load_func(**kwargs)
    
    if 'path' in kwargs:
        folder_path = kwargs['path']
    elif 'year' in kwargs and 'data_type' in kwargs and 'label_type' in kwargs:
        folder_path = f"{root_path}/{kwargs['year']}/{kwargs['data_type']}_{kwargs['label_type']}_embedding"
    try:
        emb = np.load(f'{folder_path}s.npy')
    except FileNotFoundError:
        save_in_separate(ss, folder_path)
        save_to_single(folder_path)
        emb = np.load(f'{folder_path}s.npy')
    return emb, cs, ps


def main():
    emb, cs, ps = load_embedded(load_training_data, path=f'{root_path}/training_embedding')
    print(len(emb), len(cs), len(ps))
    print(cs.count(0), cs.count(1), cs.count(2))
    print(ps.count(0), ps.count(1))


if __name__ == '__main__':
    main()
    
