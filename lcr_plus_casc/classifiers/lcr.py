"""LCR-Rot-hop++ model: two-tower (pol + cat) bilinear/hierarchical attention over BERT embeddings."""

import re

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, BertModel

from ..config import config, domain as active_domain, training

SEP_ID = 102
CLS_ID = 101


def _bert_type():
    return active_domain().bert_model


def tokenize_separated(sentence, tokenizer=None):
    """Tokenize the three ' [SEP] '-separated segments of a sentence separately.

    Returns ``(input_ids [1, L], ranges [6] long)`` where ``input_ids`` is
    ``[CLS] left [SEP] target [SEP] right [SEP]`` (left keeps its last
    ``left_max_length`` tokens, target its last ``target_max_length``, right its
    first ``right_max_length``) and ``ranges`` is
    `(l_s, l_e, t_s, t_e, r_s, r_e)` (half-open segment spans).
    """
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(_bert_type())
    parts = sentence.split(' [SEP] ')
    while len(parts) < 3:
        parts.append('')
    left_s, target_s, right_s = parts[0], parts[1], parts[2]
    if not target_s.strip():
        # No explicit target span. Plain gold rows mark the target as a
        # maximal word run joined by ' # # #' separators; fall back to the
        # last left word, then [UNK], so the target range is never empty.
        if '# # #' in sentence:
            m = re.search(r'\S+(?: # # # \S+)+', sentence)
            target_s = m.group(0) if m else '[UNK]'
        else:
            fallback = left_s.split() or right_s.split() or sentence.split()
            target_s = fallback[-1] if fallback else '[UNK]'
    left = tokenizer(left_s, add_special_tokens=False)['input_ids'][-training.left_max_length:]
    target = tokenizer(target_s, add_special_tokens=False)['input_ids'][-training.target_max_length:]
    right = tokenizer(right_s, add_special_tokens=False)['input_ids'][:training.right_max_length]
    input_ids = [CLS_ID] + left + [SEP_ID] + target + [SEP_ID] + right + [SEP_ID]
    l_s, l_e = 1, 1 + len(left)
    t_s, t_e = l_e + 1, l_e + 1 + len(target)
    r_s, r_e = t_e + 1, t_e + 1 + len(right)
    ranges = torch.tensor([l_s, l_e, t_s, t_e, r_s, r_e], dtype=torch.long)
    return torch.tensor([input_ids], dtype=torch.long), ranges


def encode_separated_batch(sentences, tokenizer=None):
    """Batch-tokenize ' [SEP] '-separated sentences at their natural lengths.

    Returns ``(input_ids [B, L], attention_mask [B, L], ranges [B, 6] long)``
    where each row of ``ranges`` is `(l_s, l_e, t_s, t_e, r_s, r_e)` and rows
    are right-padded with 0.
    """
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(_bert_type())
    rows, ranges = [], []
    for sent in sentences:
        ids, r = tokenize_separated(sent, tokenizer=tokenizer)
        rows.append(ids[0])
        ranges.append(r)
    max_len = max(config['max_length'], max(len(r) for r in rows))
    input_ids = torch.zeros((len(rows), max_len), dtype=torch.long)
    for i, row in enumerate(rows):
        input_ids[i, :len(row)] = row
    attention_mask = (input_ids != 0).long()
    return input_ids, attention_mask, torch.stack(ranges)


def masked_mean(x, mask):
    """Mask-aware mean pool across the sequence dim.

    x: [batch, L, dim]; mask: [batch, L] (1 where valid).
    """
    mask = mask.unsqueeze(-1).to(x.dtype)
    sums = (x * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1)
    return sums / counts


class BilinearAttention(nn.Module):
    """Bilinear attention over a target sequence, keyed by a pooled query."""

    def __init__(self, dim, weight_init=0.1):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(dim, dim))
        self.bias = nn.Parameter(torch.zeros(1))
        nn.init.uniform_(self.weight, -weight_init, weight_init)

    def forward(self, hidden, attention_mask=None, pool_target=None):
        # hidden: [batch, L, dim]; attention_mask: [batch, L]; pool_target: [batch, dim]
        first_term = torch.einsum('bik,bk->bi', hidden @ self.weight, pool_target)
        if attention_mask is not None:
            first_term = first_term.masked_fill(attention_mask == 0, -1e9)
        alpha = F.softmax(first_term + self.bias, dim=1)
        if attention_mask is not None:
            alpha = torch.where(attention_mask.bool(), alpha, torch.zeros_like(alpha))
        return torch.einsum('bi,bik->bk', alpha, hidden)


class HierarchicalAttention(nn.Module):
    """Attention over a stack of segment representations, returns scaled stack."""

    def __init__(self, dim, weight_init=0.1):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(dim, 1))
        self.bias = nn.Parameter(torch.zeros(1))
        nn.init.uniform_(self.weight, -weight_init, weight_init)

    def forward(self, representations):
        # representations: [batch, N, dim]
        first_term = representations @ self.weight
        alpha = F.softmax(torch.tanh(first_term + self.bias), dim=1)
        return representations * alpha


class _Branch(nn.Module):
    """One independent tower (pol or cat): BiLSTMs + bilinear & hierarchical attention."""

    def __init__(self, embedding_dim, hidden_units, invert, hierarchy):
        super().__init__()
        self.invert = invert
        self.hierarchy = hierarchy
        self.hidden_units = hidden_units

        self.left_bilstm = nn.LSTM(embedding_dim, hidden_units, bidirectional=True, batch_first=True)
        self.target_bilstm = nn.LSTM(embedding_dim, hidden_units, bidirectional=True, batch_first=True)
        self.right_bilstm = nn.LSTM(embedding_dim, hidden_units, bidirectional=True, batch_first=True)

        dim = 2 * hidden_units
        self.att_left = BilinearAttention(dim)
        self.att_target_left = BilinearAttention(dim)
        self.att_target_right = BilinearAttention(dim)
        self.att_right = BilinearAttention(dim)

        if hierarchy is not None and hierarchy[0]:
            self.hierarchical = HierarchicalAttention(dim)
        else:
            self.hierarchical_inner = HierarchicalAttention(dim)
            self.hierarchical_outer = HierarchicalAttention(dim)

    def forward(self, x_left, left_mask, x_target, target_mask, x_right, right_mask, hop):
        left = self.left_bilstm(x_left)[0]
        target = self.target_bilstm(x_target)[0]
        right = self.right_bilstm(x_right)[0]

        rep_left = masked_mean(left, left_mask)
        rep_t_left = masked_mean(target, target_mask)
        rep_t_right = masked_mean(target, target_mask)
        rep_right = masked_mean(right, right_mask)

        for _ in range(hop):
            rep_left, rep_t_left, rep_t_right, rep_right = self._bilinear(
                left, left_mask, target, target_mask, right, right_mask,
                rep_left, rep_t_left, rep_t_right, rep_right)
            if self.hierarchy is not None and self.hierarchy[1]:
                rep_left, rep_t_left, rep_t_right, rep_right = self._hierarchical(
                    rep_left, rep_t_left, rep_t_right, rep_right)

        if self.hierarchy is not None and (
            not self.hierarchy[1] or hop == 0):
            rep_left, rep_t_left, rep_t_right, rep_right = self._hierarchical(
                rep_left, rep_t_left, rep_t_right, rep_right)

        return torch.cat([rep_left, rep_t_left, rep_t_right, rep_right], dim=-1)

    def _bilinear(self, left, left_mask, target, target_mask, right, right_mask,
                  rep_left, rep_t_left, rep_t_right, rep_right):
        l = self.att_left
        tl = self.att_target_left
        tr = self.att_target_right
        r = self.att_right

        if self.invert:
            rep_t_left = tl(target, target_mask, rep_left)
            rep_t_right = tr(target, target_mask, rep_right)

        rep_left = l(left, left_mask, rep_t_left)
        rep_right = r(right, right_mask, rep_t_right)

        if not self.invert:
            rep_t_left = tl(target, target_mask, rep_left)
            rep_t_right = tr(target, target_mask, rep_right)

        return rep_left, rep_t_left, rep_t_right, rep_right

    def _hierarchical(self, rep_left, rep_t_left, rep_t_right, rep_right):
        if self.hierarchy[0]:
            reps = torch.stack([rep_left, rep_t_left, rep_t_right, rep_right], dim=1)
            rep_left, rep_t_left, rep_t_right, rep_right = torch.unbind(
                self.hierarchical(reps), dim=1)
        else:
            reps = torch.stack([rep_left, rep_right], dim=1)
            rep_left, rep_right = torch.unbind(self.hierarchical_outer(reps), dim=1)
            reps = torch.stack([rep_t_left, rep_t_right], dim=1)
            rep_t_left, rep_t_right = torch.unbind(self.hierarchical_inner(reps), dim=1)
        return rep_left, rep_t_left, rep_t_right, rep_right


def _segment_mean(h, attention_mask, s, e):
    """Masked mean over each row's [s, e) span of h.

    h: [B, L, D]; attention_mask: [B, L] (1 where valid); s, e: [B] long.
    Empty or pad-only spans yield a zero vector.
    """
    b, L, _ = h.shape
    pos = torch.arange(L, device=h.device)
    seg = (pos[None, :] >= s[:, None]) & (pos[None, :] < e[:, None])
    return masked_mean(h, (seg & attention_mask.bool()).float())


def _last_hidden(h, s, e):
    """Hidden state at index e-1 for each row (last token of the span).

    h: [B, L, D]; s, e: [B] long. Empty spans (e <= s) yield zero vectors.
    """
    b, L, d = h.shape
    valid = (e > s).float()
    idx = (e - 1).clamp(min=0, max=L - 1)
    feat = h.gather(1, idx.unsqueeze(-1).expand(b, 1, d)).squeeze(1)
    return feat * valid[:, None]


class LCRRothopPP(nn.Module):
    """Two-tower (pol + cat) LCR-Rot-hop++ model with an in-graph BERT encoder.

    ``forward(input_ids, attention_mask, ranges)`` runs BERT, pools the mean
    of the last four hidden states into a single segment feature (mode selected
    by ``embed_type``), and feeds it as a length-1 sequence to both towers.
    """

    def __init__(self, num_pol=2, num_cat=3, hidden_units=None,
                 invert=False, hop=1, hierarchy=(False, True),
                 drop_1=0.2, drop_2=0.5,
                 bert_type=None, embed_type=None):
        super().__init__()
        if bert_type is None:
            bert_type = active_domain().bert_model
        if embed_type is None:
            embed_type = training.lcr_embed_type
        self.embed_type = embed_type
        self.bert = BertModel.from_pretrained(bert_type)
        h = self.bert.config.hidden_size
        feat_dim = 3 * h if embed_type == 'combined' else h
        self.hidden_units = 768 if hidden_units is None else hidden_units
        self.hop = hop
        self.hierarchy = hierarchy

        self.branch_pol = _Branch(feat_dim, self.hidden_units, invert, hierarchy)
        self.branch_cat = _Branch(feat_dim, self.hidden_units, invert, hierarchy)
        self.drop_output = nn.Dropout(drop_2)

        # Each of the 4 reps is 2*hidden_units wide (mean over a BiLSTM output),
        # so the concatenated tower output is 8*hidden_units wide.
        tower_dim = 8 * self.hidden_units
        self.pol_dense = nn.Linear(tower_dim, num_pol)
        self.cat_dense = nn.Linear(tower_dim, num_cat)

    def forward(self, input_ids, attention_mask, ranges):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                        output_hidden_states=True)
        h = torch.stack(out.hidden_states[-4:]).mean(dim=0)
        pooled = self._pool(h, attention_mask, ranges)
        x = pooled.unsqueeze(1)
        m = torch.ones(x.shape[0], 1, dtype=torch.long, device=x.device)

        v = self.branch_pol(x, m, x, m, x, m, self.hop)
        v1 = self.branch_cat(x, m, x, m, x, m, self.hop)

        v = self.drop_output(v)
        v1 = self.drop_output(v1)

        return {'pol': self.pol_dense(v), 'cat': self.cat_dense(v1)}

    def _pool(self, h, attention_mask, ranges):
        mode = self.embed_type
        if mode == 'combined':
            left = _segment_mean(h, attention_mask, ranges[:, 0], ranges[:, 1])
            target = _segment_mean(h, attention_mask, ranges[:, 2], ranges[:, 3])
            right = _segment_mean(h, attention_mask, ranges[:, 4], ranges[:, 5])
            return torch.cat([left, target, right], dim=-1)
        if mode == 'last':
            return _last_hidden(h, ranges[:, 2], ranges[:, 3])
        if mode == 'cls':
            return h[:, 0, :]
        if mode == 'mean':
            return masked_mean(h, attention_mask)
        raise ValueError(f'unknown lcr_embed_type: {mode!r}')

    def regularization_params(self):
        """Parameters subject to L1/L2 penalties (hierarchy layers + dense heads)."""
        params = []
        for m in (self.branch_pol, self.branch_cat):
            for name in ('att_left', 'att_target_left', 'att_target_right', 'att_right'):
                params.append(getattr(m, name).weight)
                params.append(getattr(m, name).bias)
            if m.hierarchy is not None and m.hierarchy[0]:
                params.append(m.hierarchical.weight)
                params.append(m.hierarchical.bias)
            else:
                for n in ('hierarchical_inner', 'hierarchical_outer'):
                    params.append(getattr(m, n).weight)
                    params.append(getattr(m, n).bias)
        for d in (self.pol_dense, self.cat_dense):
            params.append(d.weight)
            params.append(d.bias)
        return params

    @torch.no_grad()
    def predict(self, loader, device=None, on_progress=None):
        """Run inference over ``loader`` (no gradients).

        Each batch is ``(input_ids, attention_mask, ranges, cats, pols)``; returns a
        list of ``{'cat': cat_logits, 'pol': pol_logits, 'cats': cats, 'pols': pols}``
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
            input_ids, attention_mask, ranges, cats, pols = (x.to(device) for x in batch)
            out = self(input_ids, attention_mask, ranges)
            preds.append({'cat': out['cat'].cpu(), 'pol': out['pol'].cpu(),
                          'cats': cats.cpu(), 'pols': pols.cpu()})
            samples_seen += len(cats)
            if on_progress is not None:
                on_progress(samples_seen)
        return preds
