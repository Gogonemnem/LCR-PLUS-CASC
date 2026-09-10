from ..config import *
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
            lines = f.readlines()
            for idx, line in enumerate(lines):
                if idx % 2 == 1:
                    values = line.strip().split()
                    for j in range(0, len(values), 3):
                        construct = values[j][:-1]
                        value = float(values[j+1][:-1])

                        dist[construct].append(value)

        # Compute mean and sigma for each category
        means = {}
        sigma = {}
        for key in dist:
            means[key] = np.mean(dist[key])
            sigma[key] = np.std(dist[key])

        cnt = {}
        with open(f'{self.root_path}/label.txt', 'w', encoding='utf-8') as nf:
            sentence = None
            for idx, line in enumerate(lines):
                if idx % 2 == 1:
                    aspect = []
                    aspect_word = None
                    sentiment = []
                    values = line.strip().split()

                    # Normalise score
                    for j in range(0, len(values), 3):
                        construct = values[j][:-1]
                        value = float(values[j+1][:-1])
                        # No decision possible if scores are constant
                        if sigma[construct] == 0:
                            dev = 0.0
                        else:
                            dev = (float(value) - means[construct]) / sigma[construct]

                        if dev >= lambda_threshold:
                            if construct in categories:
                                aspect.append(construct)
                                aspect_word = values[j+2]
                            else:
                                sentiment.append(construct)

                    # No conflict (avoid multi-class sentences)
                    if len(aspect) == 1 and len(sentiment) == 1:
                        separated_sentence = separate_sentence(aspect_word, sentence)
                        if separated_sentence is None:
                            continue
                        nf.write(separated_sentence)
                        nf.write(f'{aspect[0]} {sentiment[0]}\n')
                        keyword = f'{aspect[0]}-{sentiment[0]}'
                        cnt[keyword] = cnt.get(keyword, 0) + 1
                else:
                    sentence = line
        nf.close()
        # Labeled data statistics
        print('Labeled data statistics:')
        print(cnt)


def separate_sentence(pattern, sentence):
    # Escape: tokens may contain regex metacharacters (e.g. "can't", "1.5")
    match = re.search(re.escape(pattern), sentence)
    if match is None:
        # Token not found verbatim (e.g. BERT subtoken or case mismatch)
        return None
    before = sentence[:match.start()].rstrip()
    after = sentence[match.end():].lstrip()
    return f"{before} [SEP] {pattern} [SEP] {after}"
