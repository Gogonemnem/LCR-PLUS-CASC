"""Compute unnormalised overlap scores per category/polarity and write scores.txt."""
from tqdm import tqdm

from ..config import config, domain, training
from .dictionary import filter_words
from .mlm import MLMScorer


class ScoreComputer:
    """Score each sentence against per-category vocabularies via MLM top-K."""

    def __init__(self, aspect_vocabularies, sentiment_vocabularies, scorer=None):
        if scorer is None:
            scorer = MLMScorer()
        self.scorer = scorer
        self.root_path = domain().root_path
        self.aspect_vocabularies = aspect_vocabularies
        self.sentiment_vocabularies = sentiment_vocabularies

    def _header(self, categories, polarities):
        cols = ['sentence']
        for group in (categories, polarities):
            for name in group:
                cols += [f'{name}_score', f'{name}_word']
        return cols

    def __call__(self, sentences, aspects, opinions):
        cfg = domain()
        categories = cfg.categories
        polarities = cfg.polarities
        K = cfg.K_2

        aspect_sets = self._load_vocabulary(self.aspect_vocabularies, cfg.M)
        polarity_sets = self._load_vocabulary(self.sentiment_vocabularies, cfg.M)
        convert = self.scorer.tokenizer.convert_ids_to_tokens
        default = training.label_score_default
        total_cols = len(categories) + len(polarities)

        with open(f'{self.root_path}/intermediate/scores.txt', 'w', encoding='utf-8') as f:
            f.write('\t'.join(self._header(categories, polarities)) + '\n')
            for sentence, aspect, opinion in tqdm(zip(sentences, aspects, opinions)):
                if opinion == '##' or aspect == '##':
                    if training.label_z_population == 'all':
                        row = [sentence] + ['0.0', '##'] * total_cols
                        f.write('\t'.join(row) + '\n')
                    continue

                aspect_words = set(aspect.split())
                opinion_words = set(opinion.split())
                tokens, word_ids = self.scorer.topk(sentence, K, max_length=config['max_length'])

                if training.label_agg == 'sum':
                    cat_scores, cat_words = self._aggregate_sum(
                        tokens, word_ids, convert, aspect_words, aspect_sets, categories, default)
                    pol_scores, pol_words = self._aggregate_sum(
                        tokens, word_ids, convert, opinion_words, polarity_sets, polarities, default)
                else:
                    cat_scores, cat_words = self._aggregate_max(
                        tokens, word_ids, convert, aspect_words, aspect_sets, categories, default)
                    pol_scores, pol_words = self._aggregate_max(
                        tokens, word_ids, convert, opinion_words, polarity_sets, polarities, default)

                row = [sentence]
                for cat in categories:
                    row += [str(cat_scores[cat]), cat_words[cat]]
                for pol in polarities:
                    row += [str(pol_scores[pol]), pol_words[pol]]
                f.write('\t'.join(row) + '\n')

    def _aggregate_max(self, tokens, word_ids, convert, target_words, sets, classes, default):
        """Paper mode: per-class score = max over matched tokens of (#top-K replacements in set)."""
        scores = {c: default for c in classes}
        words = {c: '##' for c in classes}
        for idx, token in enumerate(tokens):
            if token not in target_words:
                continue
            replacements = convert(word_ids[idx])
            for cls in classes:
                score = sum(
                    1 for repl in replacements
                    if repl not in filter_words and '##' not in repl
                    and repl in sets[cls]
                )
                if score > scores[cls]:
                    scores[cls] = score
                    words[cls] = token
        return scores, words

    def _aggregate_sum(self, tokens, word_ids, convert, target_words, sets, classes, default):
        """Reference mode: each matched top-K replacement contributes +1 to the first class whose
        set contains it; final score = total / max(#matched-token occurrences, 1)."""
        totals = {c: 0 for c in classes}
        token_hits = {c: {} for c in classes}
        occurrences = 0
        for idx, token in enumerate(tokens):
            if token not in target_words:
                continue
            occurrences += 1
            for repl in convert(word_ids[idx]):
                if repl in filter_words or '##' in repl:
                    continue
                for cls in classes:
                    if repl in sets[cls]:
                        totals[cls] += 1
                        token_hits[cls][token] = token_hits[cls].get(token, 0) + 1
                        break
        scores = {}
        words = {}
        for cls in classes:
            if totals[cls] == 0:
                scores[cls] = default
                words[cls] = '##'
            else:
                scores[cls] = totals[cls] / max(occurrences, 1)
                words[cls] = max(token_hits[cls], key=token_hits[cls].get)
        return scores, words

    def _load_vocabulary(self, source, limit):
        return {key: {word for _, word in source[key][:limit]} for key in source}
