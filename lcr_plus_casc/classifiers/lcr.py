"""LCR-Rot-hop++ model: two-tower (pol + cat) bilinear/hierarchical attention over BERT embeddings."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import left_max_length, right_max_length, target_max_length


class BilinearAttention(nn.Module):
    """Bilinear attention over a target sequence, keyed by a pooled query."""

    def __init__(self, dim, weight_init=0.1):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(dim, dim))
        self.bias = nn.Parameter(torch.zeros(1))
        nn.init.uniform_(self.weight, -weight_init, weight_init)

    def forward(self, hidden, pool_target):
        # hidden: [batch, L, dim]; pool_target: [batch, dim]
        first_term = torch.einsum('bik,bk->bi', hidden @ self.weight, pool_target)
        alpha = F.softmax(torch.tanh(first_term + self.bias), dim=1)
        return torch.einsum('bki,bk->bi', hidden, alpha)


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

    def forward(self, x_left, x_target, x_right, hop):
        left = self.left_bilstm(x_left)[0]
        target = self.target_bilstm(x_target)[0]
        right = self.right_bilstm(x_right)[0]

        rep_left = left.mean(1)
        rep_t_left = target.mean(1)
        rep_t_right = target.mean(1)
        rep_right = right.mean(1)

        for _ in range(hop):
            rep_left, rep_t_left, rep_t_right, rep_right = self._bilinear(
                left, target, right, rep_left, rep_t_left, rep_t_right, rep_right)
            if self.hierarchy is not None and self.hierarchy[1]:
                rep_left, rep_t_left, rep_t_right, rep_right = self._hierarchical(
                    rep_left, rep_t_left, rep_t_right, rep_right)

        if self.hierarchy is not None and (
            not self.hierarchy[1] or hop == 0):
            rep_left, rep_t_left, rep_t_right, rep_right = self._hierarchical(
                rep_left, rep_t_left, rep_t_right, rep_right)

        return torch.cat([rep_left, rep_t_left, rep_t_right, rep_right], dim=-1)

    def _bilinear(self, left, target, right, rep_left, rep_t_left, rep_t_right, rep_right):
        l = self.att_target_left
        tl = self.att_target_left
        tr = self.att_target_right
        r = self.att_right

        if self.invert:
            rep_t_left = tl(target, rep_left)
            rep_t_right = tr(target, rep_right)

        rep_left = l(left, rep_t_left)
        rep_right = r(right, rep_t_right)

        if not self.invert:
            rep_t_left = tl(target, rep_left)
            rep_t_right = tr(target, rep_right)

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


class LCRRothopPP(nn.Module):
    """Two-tower (pol + cat) LCR-Rot-hop++ model ported from the TF reference."""

    def __init__(self, embedding_dim=768, num_pol=2, num_cat=3, hidden_units=None,
                 invert=False, hop=1, hierarchy=(False, True),
                 drop_1=0.2, drop_2=0.5, l1=0.0, l2=0.0):
        super().__init__()
        self.hidden_units = 768 if hidden_units is None else hidden_units
        self.hop = hop
        self.hierarchy = hierarchy
        self.l1 = l1
        self.l2 = l2

        self.branch_pol = _Branch(embedding_dim, self.hidden_units, invert, hierarchy)
        self.branch_cat = _Branch(embedding_dim, self.hidden_units, invert, hierarchy)
        self.drop_output = nn.Dropout(drop_2)

        # Each of the 4 reps is 2*hidden_units wide (mean over a BiLSTM output),
        # so the concatenated tower output is 8*hidden_units wide.
        tower_dim = 8 * self.hidden_units
        self.pol_dense = nn.Linear(tower_dim, num_pol)
        self.cat_dense = nn.Linear(tower_dim, num_cat)

    def forward(self, inputs):
        # inputs: [batch, total_seq, embedding_dim]
        x_left = inputs[:, 1:left_max_length + 1]
        x_target = inputs[:, left_max_length + 2:left_max_length + target_max_length + 2]
        x_right = inputs[:, left_max_length + target_max_length + 3:
                         left_max_length + target_max_length + right_max_length + 3]

        v = self.branch_pol(x_left, x_target, x_right, self.hop)
        v1 = self.branch_cat(x_left, x_target, x_right, self.hop)

        v = self.drop_output(v)
        v1 = self.drop_output(v1)

        return {'pol': self.pol_dense(v), 'cat': self.cat_dense(v1)}

    @property
    def _regularized_params(self):
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

    def regularization_loss(self):
        loss = torch.zeros((), device=next(self.parameters()).device)
        for p in self._regularized_params:
            if self.l1:
                loss = loss + self.l1 * p.abs().sum()
            if self.l2:
                loss = loss + self.l2 * (p ** 2).sum()
        return loss
