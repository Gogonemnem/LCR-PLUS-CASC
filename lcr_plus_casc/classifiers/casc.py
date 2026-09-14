"""CASC model: BERT sentence encoder with independent category/polarity heads."""

import torch
from torch import nn
from transformers import BertModel


class BERTLinear(nn.Module):
    """BERT classifier with linear aspect-category and polarity heads."""

    def __init__(self, bert_type, num_cat, num_pol):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_type, output_hidden_states=True)
        hidden_size = self.bert.config.hidden_size
        self.ff_cat = nn.Linear(hidden_size, num_cat)
        self.ff_pol = nn.Linear(hidden_size, num_pol)

    def forward(self, input_ids, attention_mask):
        """Returns ``{'cat': logits, 'pol': logits}``."""
        bert_hidden = self.bert(input_ids=input_ids, attention_mask=attention_mask).hidden_states[-1]
        mask = attention_mask.unsqueeze(-1).to(bert_hidden.dtype)  # (bsz, seq_len, 1)
        sentence_emb = (bert_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return {'cat': self.ff_cat(sentence_emb), 'pol': self.ff_pol(sentence_emb)}

    def regularization_params(self):
        """Parameters subject to L1/L2 penalties: the two linear heads."""
        yield from self.ff_cat.parameters()
        yield from self.ff_pol.parameters()

    @torch.no_grad()
    def predict(self, loader, device=None, on_progress=None):
        """Run inference over ``loader`` (no gradients).

        Each batch is ``(input_ids, attention_mask, cats, pols)``; returns a list
        of ``{'cat': cat_logits, 'pol': pol_logits, 'cats': cats, 'pols': pols}``
        (one entry per batch), all on CPU. Consumed by the shared
        :func:`~lcr_plus_casc.losses.evaluate`.

        ``on_progress`` (optional) is called once per batch with the cumulative
        number of samples evaluated so far in this call (after the first batch).
        """
        if device is None:
            device = next(self.parameters()).device
        self.eval()
        preds = []
        samples_seen = 0
        for batch in loader:
            input_ids, attention_mask, cats, pols = (x.to(device) for x in batch)
            out = self(input_ids, attention_mask)
            preds.append({'cat': out['cat'].cpu(), 'pol': out['pol'].cpu(),
                          'cats': cats.cpu(), 'pols': pols.cpu()})
            samples_seen += len(cats)
            if on_progress is not None:
                on_progress(samples_seen)
        return preds
