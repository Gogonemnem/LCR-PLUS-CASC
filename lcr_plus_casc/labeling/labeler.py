"""Turn raw overlap scores into weakly-supervised (aspect, polarity) labels."""
import re

import numpy as np

from ..config import domain, training
from .split_file import read_rows


class Labeler:
    """Z-score each score column and keep sentences with a single clear label."""

    def __init__(self):
        self.root_path = domain().root_path

    def __call__(self):
        cfg = domain()
        categories = list(cfg.categories)
        polarities = list(cfg.polarities)

        rows = read_rows(f'{self.root_path}/intermediate/scores.txt')

        # column layout: [sentence, {cat}_score, {cat}_word, ..., {pol}_score, {pol}_word]
        cols = []
        i = 1
        for cat in categories:
            cols.append({'name': cat, 'score': i, 'word': i + 1, 'aspect': True})
            i += 2
        for pol in polarities:
            cols.append({'name': pol, 'score': i, 'word': i + 1, 'aspect': False})
            i += 2

        dist = {c['name']: [] for c in cols}
        for row in rows:
            for c in cols:
                dist[c['name']].append(float(row[c['score']]))

        means = {name: float(np.mean(values)) for name, values in dist.items()}
        sigma = {name: float(np.std(values)) for name, values in dist.items()}

        cnt = {}
        with open(f'{self.root_path}/label.txt', 'w', encoding='utf-8') as nf:
            for idx, row in enumerate(rows):
                sentence = row[0]
                aspect = []
                aspect_word = None
                sentiment = []
                for c in cols:
                    value = float(row[c['score']])
                    s = sigma[c['name']]
                    dev = 0.0 if s == 0 else (value - means[c['name']]) / s
                    if dev >= cfg.lambda_threshold:
                        if c['aspect']:
                            aspect.append(c['name'])
                            aspect_word = row[c['word']]
                        else:
                            sentiment.append(c['name'])

                if len(aspect) == 1 and len(sentiment) == 1:
                    separated = separate_sentence(aspect_word, sentence)
                    if separated is None:
                        if training.label_require_word:
                            continue
                        separated = sentence
                    nf.write(f'{idx}\t{aspect[0]}\t{sentiment[0]}\t{separated}\n')
                    keyword = f'{aspect[0]}-{sentiment[0]}'
                    cnt[keyword] = cnt.get(keyword, 0) + 1

        print('Labeled data statistics:')
        print(cnt)


def separate_sentence(pattern, sentence):
    pattern = pattern.removeprefix('##')
    match = re.search(r'(?<!\w)' + re.escape(pattern) + r'(?!\w)', sentence)
    if match is None:
        return None
    before = sentence[:match.start()].rstrip()
    after = sentence[match.end():].lstrip()
    return f"{before} [SEP] {pattern} [SEP] {after}"
