import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertModel

BERT_HIDDEN_SIZE = 768


class LQLoss(nn.Module):
    """Generalized cross-entropy (GCE) loss with per-class weights."""

    def __init__(self, q, weight, alpha=0.0):
        super().__init__()
        self.q = q
        self.alpha = alpha
        weight = torch.log(1 / torch.as_tensor(weight, dtype=torch.float32))
        self.register_buffer('weight', F.softmax(weight, dim=-1))

    def forward(self, input, target):
        yq = torch.gather(input, 1, target.unsqueeze(1))
        lq = (1 - yq ** self.q) / self.q
        weight = torch.gather(self.weight.expand_as(input), 1, target.unsqueeze(1))
        return (self.alpha * lq + (1 - self.alpha) * lq * weight).mean()

    def set_weights(self, weights):
        weight = torch.log(1 / torch.as_tensor(weights, dtype=torch.float32, device=self.weight.device))
        self.weight.copy_(F.softmax(weight, dim=-1))


class BERTLinear(nn.Module):
    def __init__(self, bert_type, num_cat, num_pol, aspect_weights=None, sentiment_weights=None):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_type, output_hidden_states=True)
        self.ff_cat = nn.Linear(BERT_HIDDEN_SIZE, num_cat)
        self.ff_pol = nn.Linear(BERT_HIDDEN_SIZE, num_pol)
        if aspect_weights is None:
            aspect_weights = [1] * num_cat
        if sentiment_weights is None:
            sentiment_weights = [1] * num_pol
        self.loss_cat = LQLoss(0.4, aspect_weights)
        self.loss_pol = LQLoss(0.4, sentiment_weights)

    def forward(self, labels_cat, labels_pol, **kwargs):
        bert_hidden = self.bert(**kwargs).hidden_states[-1]  # (bsz, seq_len, 768)
        mask = kwargs['attention_mask'].unsqueeze(-1).to(bert_hidden.dtype)  # (bsz, seq_len, 1)
        sentence_emb = (bert_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        logits_cat = self.ff_cat(sentence_emb)
        logits_pol = self.ff_pol(sentence_emb)
        loss = (self.loss_cat(F.softmax(logits_cat, dim=-1), labels_cat)
                + self.loss_pol(F.softmax(logits_pol, dim=-1), labels_pol))
        return loss, logits_cat, logits_pol
