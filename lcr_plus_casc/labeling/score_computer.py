"""Compute unnormalised overlap scores per category/polarity and write scores.txt."""
from tqdm import tqdm

from ..config import (
    K_2,
    M,
    aspect_category_mapper,
    config,
    max_length,
    path_mapper,
    sentiment_category_mapper,
)
from .dictionary import filter_words
from .mlm import MLMScorer


class ScoreComputer:
    """Score each sentence against per-category vocabularies via MLM top-K."""

    def __init__(self, aspect_vocabularies, sentiment_vocabularies, scorer=None):
        if scorer is None:
            scorer = MLMScorer()
        self.scorer = scorer
        self.root_path = path_mapper[config['domain']]
        self.aspect_vocabularies = aspect_vocabularies
        self.sentiment_vocabularies = sentiment_vocabularies

    def _header(self, categories, polarities):
        cols = ['sentence']
        for group in (categories, polarities):
            for name in group:
                cols += [f'{name}_score', f'{name}_word']
        return cols

    def __call__(self, sentences, aspects, opinions):
        domain = config['domain']
        categories = aspect_category_mapper[domain]
        polarities = sentiment_category_mapper[domain]
        K = K_2

        aspect_sets = self._load_vocabulary(self.aspect_vocabularies, M[domain])
        polarity_sets = self._load_vocabulary(self.sentiment_vocabularies, M[domain])
        convert = self.scorer.tokenizer.convert_ids_to_tokens

        with open(f'{self.root_path}/scores.txt', 'w', encoding='utf-8') as f:
            f.write('\t'.join(self._header(categories, polarities)) + '\n')
            for sentence, aspect, opinion in tqdm(zip(sentences, aspects, opinions)):
                if opinion == '##' or aspect == '##':
                    continue

                aspect_words = set(aspect.split())
                opinion_words = set(opinion.split())

                cat_scores = {cat: -1 for cat in categories}
                cat_words = {cat: '##' for cat in categories}
                pol_scores = {pol: -1 for pol in polarities}
                pol_words = {pol: '##' for pol in polarities}

                tokens, word_ids = self.scorer.topk(sentence, K, max_length=max_length)

                for idx, token in enumerate(tokens):
                    if token in aspect_words:
                        replacements = convert(word_ids[idx])
                        for cat in categories:
                            score = sum(
                                1 for repl in replacements
                                if repl not in filter_words and '##' not in repl
                                and repl in aspect_sets[cat]
                            )
                            if score > cat_scores[cat]:
                                cat_scores[cat] = score
                                cat_words[cat] = token

                    if token in opinion_words:
                        replacements = convert(word_ids[idx])
                        for pol in polarities:
                            score = sum(
                                1 for repl in replacements
                                if repl not in filter_words and '##' not in repl
                                and repl in polarity_sets[pol]
                            )
                            if score > pol_scores[pol]:
                                pol_scores[pol] = score
                                pol_words[pol] = token

                row = [sentence]
                for cat in categories:
                    row += [str(cat_scores[cat]), cat_words[cat]]
                for pol in polarities:
                    row += [str(pol_scores[pol]), pol_words[pol]]
                f.write('\t'.join(row) + '\n')

    def _load_vocabulary(self, source, limit):
        return {key: {word for _, word in source[key][:limit]} for key in source}
