from logging import root
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


def load_training_data(**kwargs):
    sentences = []
    cats = []
    pols = []
    if 'training_path' in kwargs:
        training_path = kwargs['training_path']
    else:
        training_path = f'{root_path}/label.txt'
    with open(training_path, 'r', encoding='utf-8') as f:
        skip = False
        for idx, line in enumerate(f):
            # if skip:
            #     skip = False
            #     continue

            if idx % 2 == 1:
                cat, pol = line.strip().split()
                cats.append(inv_aspect_dict[cat])
                pols.append(inv_polarity_dict[pol])
            else:
                # if line[0:2] == "##":
                #     skip = True
                #     continue
                sentences.append(line.strip())
    return sentences, cats, pols

def load_training_data2(**kwargs):
    sentences = []
    cats = []
    pols = []
    if 'training_path' in kwargs:
        training_path = kwargs['training_path']
        if '2015' in training_path:
            p = r'datasets\restaurant\2015\test_single.txt'
        elif '2016' in training_path:
            p = r'datasets\restaurant\2016\test_single.txt'
    else:
        training_path = f'{root_path}/label.txt'
    # with open(training_path, 'r', encoding='utf-8') as f:
    with open(training_path, 'r', encoding='utf-8') as f, open(p, 'r', encoding='utf-8') as f1:
        skip = False
        skips = []
        for idx, line in enumerate(f):
            if skip:
                skip = False
                continue

            if idx % 2 == 1:
                continue
                # cat, pol = line.strip().split()
                # cats.append(inv_aspect_dict[cat])
                # pols.append(inv_polarity_dict[pol])
            else:
                if line[0:2] == "##":
                    skip = True
                    skips.append(idx/2)
                    continue
                sentences.append(line.strip())

        for idx, line in enumerate(f1):
            if idx in skips:
                continue

            split_line = line.strip().split('\t')
            if len(split_line) < 4:
                continue

            _, cat, pol, sentence = split_line
            cats.append(int(cat))
            pols.append(int(pol))

    return sentences, cats, pols

def save_in_separate(sentences, folder_path):
    os.makedirs(folder_path, exist_ok=True)
    total = len(str(len(sentences)))

    
    for i, sent in enumerate(sentences):
        # if i <=74:
        #     continue
        # if i >= 1000:
        #     break

        print(sent)
        tokens = tokenize_separated(sent)
        embedding = embed_separated(tokens)
    
        # print(tokens)
        print(embedding)
        print(i)
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
    folder_path = f'{root_path}/training_embedding'
    # save_in_separate(ss, folder_path)
    folder_path = f'{root_path}/training_embedding/'
    # save_in_single(folder_path)
    emb = np.load(f'{root_path}/training_embeddings.npy')
    return emb, cs, ps

# dt = tf.data.Dataset.from_tensor_slices((emb, cs, ps))
def load_semeval(year, data_type, label_type, **kwargs):
    with open(f'{root_path}/{str(year)}/{data_type}_{label_type}.txt', 'r', encoding='utf-8') as f:
        sentences = []
        cats = []
        pols = []

        for line in f:
            split_line = line.strip().replace(' [SEP] ', '').split('\t')
            if len(split_line) < 4:
                continue

            _, cat, pol, sentence = split_line
            cats.append(int(cat))
            pols.append(int(pol))
            sentences.append(sentence)
        
        return sentences, cats, pols
    
# print(load_semeval())

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
    # print(cs)
    # print(ps)
    # emb, cs, ps = load_embedded(load_training_data2, path=f'{root_path}/test_embedding_2016', training_path=r'datasets\restaurant\label 2016 single.txt')
    # print(len(emb), len(cs), len(ps))

    # emb, cs, ps = load_embedded(load_semeval, year=2015, data_type='test', label_type='single')
    # print(len(emb), len(cs), len(ps))
    # print(cs.count(0), cs.count(1), cs.count(2))
    # print(ps.count(0), ps.count(1))
    # # load_embedded(load_semeval, year=2015, data_type='val', label_type='single')

    # emb, cs, ps = load_embedded(load_semeval, year=2015, data_type='test', label_type='multi')
    # print(len(emb), len(cs), len(ps))
    # print(cs.count(0), cs.count(1), cs.count(2))
    # print(ps.count(0), ps.count(1))
    # # load_embedded(load_semeval, year=2015, data_type='val', label_type='multi')

    # emb, cs, ps = load_embedded(load_semeval, year=2016, data_type='test', label_type='single')
    # print(len(emb), len(cs), len(ps))
    # print(cs.count(0), cs.count(1), cs.count(2))
    # print(ps.count(0), ps.count(1))
    # # load_embedded(load_semeval, year=2016, data_type='val', label_type='single')

    # emb, cs, ps = load_embedded(load_semeval, year=2016, data_type='test', label_type='multi')
    # print(len(emb), len(cs), len(ps))
    # print(cs.count(0), cs.count(1), cs.count(2))
    # print(ps.count(0), ps.count(1))
    # load_embedded(load_semeval, year=2016, data_type='val', label_type='multi')


if __name__ == '__main__':
    main()
    
