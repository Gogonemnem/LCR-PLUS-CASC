"""Extract candidate aspect and opinion words from raw sentences with spaCy."""
from tqdm import tqdm
import spacy

from ..config import config, path_mapper, spacy_model


class Extracter:
    """Tag POS candidates (opinion: JJ/RR, aspect: NN*) from each sentence."""

    def __init__(self):
        self.smodel = spacy.load(spacy_model)
        self.root_path = path_mapper[config['domain']]

    def extract(self, doc):
        """Return (opinion_words, aspect_words) from a spaCy Doc."""
        o = [word.text for word in doc if word.tag_[:2] in ('JJ', 'RR')]
        a = [word.text for word in doc if word.tag_.startswith('NN')]
        return o, a

    def __call__(self):
        with open(f'{self.root_path}/train.txt', encoding='utf-8') as f:
            sentences = [line.strip() for line in f]

        aspects = []
        opinions = []
        for doc in tqdm(self.smodel.pipe(sentences), total=len(sentences)):
            o, a = self.extract(doc)
            opinions.append(' '.join(o) if o else '##')
            aspects.append(' '.join(a) if a else '##')

        return sentences, aspects, opinions
