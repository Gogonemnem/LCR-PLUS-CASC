import os

from config import *
from embedding import embed_separated, tokenize_separated


import numpy as np
import tensorflow as tf


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

print(polarity_dict, aspect_dict)


def load_training_data():
    sentences = []
    cats = []
    pols = []
    with open(f'{root_path}/label.txt', 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if idx % 2 == 1:
                cat, pol = line.strip().split()
                cats.append(inv_aspect_dict[cat])
                pols.append(inv_polarity_dict[pol])
            else:
                sentences.append(line.strip())
    return sentences, cats, pols

def save_in_separate(sentences):
    os.makedirs(f'{root_path}/training_embedding', exist_ok=True)
    total = len(str(len(sentences)))

    
    for i, sent in enumerate(sentences):
        if i <=74:
            continue
        if i >= 1000:
            break

        print(sent)
        tokens = tokenize_separated(sent)
        embedding = embed_separated(tokens)
    
        # print(tokens)
        print(embedding)
        print(i)
        number = str(i).rjust(total, '0')

        

        np.save(f'{root_path}/training_embedding/{number}', embedding)

def save_to_single():
    folder_path = f'{root_path}/training_embedding/'
    files = os.listdir(folder_path)

    first = np.load(f'{folder_path}{files[0]}')
    all_embeddings = np.zeros((len(files), first.shape[1], first.shape[2]))

    for i, file in enumerate(files):
        array = np.load(f'{folder_path}{file}')
        all_embeddings[i, :, :] = array[0, :, :]

    np.save(f'{root_path}/training_embeddings', all_embeddings)

def load_embedded_training():
    ss, cs, ps = load_training_data()
    # save_in_separate(ss)
    # save_in_single()
    emb = np.load(f'{root_path}/training_embeddings.npy')
    return emb, cs, ps

# dt = tf.data.Dataset.from_tensor_slices((emb, cs, ps))
