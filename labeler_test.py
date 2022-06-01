from config import *
import numpy as np
import re

class Labeler:

    def __init__(self):
        self.domain = config['domain']
        self.root_path = path_mapper[self.domain]
    
    def __call__(self):
        categories = aspect_category_mapper[self.domain]
        polarities = sentiment_category_mapper[self.domain]

        # Distributions
        dist = {}
        for cat in categories:
            dist[cat] = []
        for pol in polarities:
            dist[pol] = []

        # Read scores
        with open(f'{self.root_path}/scores.txt', 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if idx % 2 == 1:
                    values = line.strip().split()
                    for j in range(0, len(values), 3):
                        construct = values[j][:-1]
                        value = float(values[j+1][:-1])
                        
                        dist[construct].append(value)
                    # for j, val in enumerate(values):
                    #     if j % 2 == 1:
                    #         dist[values[j-1][:-1]].append(float(val))
        
        # Compute mean and sigma for each category
        means = {}
        sigma = {}
        for key in dist:
            means[key] = np.mean(dist[key])
            sigma[key] = np.std(dist[key])
        
        nf = open(f'{self.root_path}/label.txt', 'w', encoding='utf-8')
        cnt = {}
        with open(f'{self.root_path}/scores.txt', 'r', encoding='utf-8') as f:
            sentence = None
            for idx, line in enumerate(f):
                if idx % 2 == 1:
                    aspect = []
                    aspect_word = None
                    sentiment = []
                    key = None
                    values = line.strip().split()

                    # Normalise score
                    for j in range(0, len(values), 3):
                        construct = values[j][:-1]
                        value = float(values[j+1][:-1])
                        dev = (float(value) - means[construct]) / sigma[construct]
                        # print(construct, dev)

                        if dev >= lambda_threshold:
                            if construct in categories:
                                aspect.append(construct)
                                aspect_word = values[j+2]
                            else:
                                sentiment.append(construct)

                    # for j, val in enumerate(values):
                    #     if j % 2 == 1:
                    #         # Normalise score
                    #         dev = (float(val) - means[key]) / sigma[key]
                    #         if dev >= lambda_threshold:
                    #             if key in categories:
                    #                 aspect.append(key)
                    #             else:
                    #                 sentiment.append(key)
                    #     else:
                    #         key = val[:-1]
                    # No conflict (avoid multi-class sentences)
                    # print(len(aspect), len(sentiment))
                    if len(aspect) == 1 and len(sentiment) == 1:
                        separated_sentence = separate_sentence(aspect_word, sentence)
                        nf.write(separated_sentence)
                        nf.write(f'{aspect[0]} {sentiment[0]}\n')
                        keyword = f'{aspect[0]}-{sentiment[0]}'
                        cnt[keyword] = cnt.get(keyword, 0) + 1
                else:
                    sentence = line
                
                # if idx>10:
                #     break
        nf.close()
        # Labeled data statistics
        print('Labeled data statistics:')
        print(cnt)


def separate_sentence(pattern, sentence):
    split_sent = re.split(pattern, sentence, maxsplit=1)
    separated_sentence = f"{split_sent[0]} [SEP] {pattern} [SEP] {split_sent[1]}"
    return separated_sentence