"""Extract candidate aspect and opinion words from raw sentences with spaCy."""
import logging

from tqdm import tqdm
import spacy

from ..config import domain


class Extracter:
    """Tag POS candidates (opinion: JJ/RR, aspect: NN*) from each sentence."""

    def __init__(self):
        try:
            self.smodel = spacy.load(domain().spacy_model)
        except OSError:
            self.smodel = None
            for model in ('en_core_web_sm', 'en_core_web_md'):
                try:
                    logging.getLogger(__name__).warning(
                        'spaCy model %r unavailable, falling back to %r',
                        domain().spacy_model, model)
                    self.smodel = spacy.load(model)
                    break
                except OSError:
                    continue
            if self.smodel is None:
                raise RuntimeError(
                    "no spaCy model available (tried %r, en_core_web_sm, en_core_web_md); "
                    'install one with: python -m spacy download en_core_web_sm'
                    % domain().spacy_model)
        self.root_path = domain().root_path

    def extract(self, doc):
        """Return (opinion_words, aspect_words) from a spaCy Doc."""
        o = [word.text for word in doc if word.tag_[:2] in ('JJ', 'RR')]
        a = [word.text for word in doc if word.tag_.startswith('NN')]
        return o, a

    def __call__(self):
        with open(f'{self.root_path}/raw/train.txt', encoding='utf-8') as f:
            sentences = [line.strip() for line in f]

        aspects = []
        opinions = []
        for doc in tqdm(self.smodel.pipe(sentences), total=len(sentences)):
            o, a = self.extract(doc)
            opinions.append(' '.join(o) if o else '##')
            aspects.append(' '.join(a) if a else '##')

        return sentences, aspects, opinions
