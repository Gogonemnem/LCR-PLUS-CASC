"""Vocabulary generator: derive per-category word lists from MLM predictions."""
from tqdm import tqdm

from ..config import (
    K_1,
    aspect_category_mapper,
    aspect_seed_mapper,
    config,
    path_mapper,
    sentiment_category_mapper,
    sentiment_seed_mapper,
)
from .dictionary import filter_words
from .mlm import MLMScorer


class VocabGenerator:
    """Build per-category vocabularies from MLM top-1 predictions on seeds."""

    def __init__(self, save_results=True, scorer=None):
        if scorer is None:
            scorer = MLMScorer()
        self.scorer = scorer
        self.domain = config['domain']
        self.root_path = path_mapper[self.domain]
        self.save_results = save_results

    def __call__(self):
        aspect_categories = aspect_category_mapper[self.domain]
        aspect_seeds = aspect_seed_mapper[self.domain]
        aspect_vocabularies = self.generate_vocabularies(aspect_categories, aspect_seeds)

        sentiment_categories = sentiment_category_mapper[self.domain]
        sentiment_seeds = sentiment_seed_mapper[self.domain]
        sentiment_vocabularies = self.generate_vocabularies(sentiment_categories, sentiment_seeds)

        return aspect_vocabularies, sentiment_vocabularies

    def generate_vocabularies(self, categories, seeds):
        freq_table = {cat: {} for cat in categories}

        for category in categories:
            print(f'Generating vocabulary for {category} category...')
            with open(f'{self.root_path}/train.txt', encoding='utf-8') as f:
                for line in tqdm(f):
                    text = line.strip()
                    if category not in text:
                        continue
                    tokens, word_ids = self.scorer.topk(text, K_1)
                    for idx, token in enumerate(tokens):
                        if token in seeds[category]:
                            self.update_table(
                                freq_table, category,
                                self.scorer.tokenizer.convert_ids_to_tokens(word_ids[idx]),
                            )

        best_cat, best_freq = {}, {}
        for category in categories:
            for word, freq in freq_table[category].items():
                if best_freq.get(word, -1) < freq:
                    best_cat[word] = category
                    best_freq[word] = freq
        for category in categories:
            freq_table[category] = {
                word: freq for word, freq in freq_table[category].items()
                if best_cat[word] == category
            }

        vocabularies = {}
        for category in categories:
            words = sorted(((freq, word) for word, freq in freq_table[category].items()),
                           reverse=True)
            vocabularies[category] = words

            if self.save_results:
                with open(f'{self.root_path}/dict_{category}.txt', 'w', encoding='utf-8') as f:
                    for freq, word in words:
                        f.write(f'{word} {freq}\n')

        return vocabularies

    def update_table(self, freq_table, cat, tokens):
        for token in tokens:
            if token in filter_words or '##' in token:
                continue
            freq_table[cat][token] = freq_table[cat].get(token, 0) + 1

    def from_folder(self, folder_path=None, aspect_categories=None, sentiment_categories=None):
        if folder_path is None:
            folder_path = self.root_path

        if aspect_categories is None:
            aspect_categories = aspect_category_mapper[self.domain]
        aspect_vocabularies = self._load_vocabulary(aspect_categories, folder_path)

        if sentiment_categories is None:
            sentiment_categories = sentiment_category_mapper[self.domain]
        sentiment_vocabularies = self._load_vocabulary(sentiment_categories, folder_path)

        return aspect_vocabularies, sentiment_vocabularies

    def _load_vocabulary(self, categories, folder_path):
        vocabularies = {}
        for category in categories:
            words = []
            with open(f'{folder_path}/dict_{category}.txt', encoding='utf-8') as f:
                for line in tqdm(f):
                    word, freq = line.strip().split()
                    words.append((int(freq), word))
            vocabularies[category] = words
        return vocabularies
