"""Shared Masked-LM scorer used by the vocabulary generator and score computer."""
import torch
from transformers import AutoTokenizer, BertForMaskedLM

from ..config import bert_mapper, config


class MLMScorer:
    def __init__(self):
        self.bert_type = bert_mapper[config['domain']]
        self.device = config['device']
        self.tokenizer = AutoTokenizer.from_pretrained(self.bert_type)
        self._model = None

    @property
    def model(self):
        if self._model is None:
            self._model = BertForMaskedLM.from_pretrained(self.bert_type).to(self.device)
        return self._model

    def tokenize(self, text, max_length=None):
        if max_length is None:
            return self.tokenizer(text, return_tensors='pt', truncation=True)['input_ids']
        return self.tokenizer(text, return_tensors='pt', truncation=True, max_length=max_length)['input_ids']

    def topk(self, sentence, k, max_length=None):
        """Predict the top-k replacement tokens per position.

        Returns (tokens, word_ids) where `tokens` are the input subtokens and
        word_ids[i] is the [k] index tensor predicting position i.
        """
        ids = self.tokenize(sentence, max_length=max_length)
        tokens = self.tokenizer.convert_ids_to_tokens(ids[0])
        with torch.no_grad():
            predictions = self.model(ids.to(self.device))[0]
        word_scores, word_ids = torch.topk(predictions, k, -1)
        return tokens, word_ids.squeeze(0)
