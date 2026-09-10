"""CASC model: BERT sentence encoder with independent category/polarity heads."""

import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertModel

from ..losses import LQLoss


class BERTLinear(nn.Module):
    """BERT classifier with linear aspect-category and polarity heads."""

    def __init__(self, bert_type, num_cat, num_pol, q, aspect_weights=None, sentiment_weights=None):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_type, output_hidden_states=True)
        hidden_size = self.bert.config.hidden_size
        self.ff_cat = nn.Linear(hidden_size, num_cat)
        self.ff_pol = nn.Linear(hidden_size, num_pol)
        if aspect_weights is None:
            aspect_weights = [1] * num_cat
        if sentiment_weights is None:
            sentiment_weights = [1] * num_pol
        self.loss_cat = LQLoss(q, aspect_weights)
        self.loss_pol = LQLoss(q, sentiment_weights)

    def forward(self, labels_cat, labels_pol, **kwargs):
        """Returns {'loss', 'logits_cat', 'logits_pol'}; kwargs pass through to BERT."""
        bert_hidden = self.bert(**kwargs).hidden_states[-1]
        mask = kwargs['attention_mask'].unsqueeze(-1).to(bert_hidden.dtype)  # (bsz, seq_len, 1)
        sentence_emb = (bert_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        logits_cat = self.ff_cat(sentence_emb)
        logits_pol = self.ff_pol(sentence_emb)
        loss = (self.loss_cat(F.softmax(logits_cat, dim=-1), labels_cat)
                + self.loss_pol(F.softmax(logits_pol, dim=-1), labels_pol))
        return loss, logits_cat, logits_pol
