"""Shared loss functions for the CASC and LCR classifiers."""
import torch
from torch import nn
import torch.nn.functional as F


class LQLoss(nn.Module):
    """Generalized cross-entropy (GCE) loss with per-class weights, applied to probabilities."""

    def __init__(self, q, weight, alpha=0.0):
        super().__init__()
        self.q = q
        self.alpha = alpha
        weight = torch.log(1 / torch.as_tensor(weight, dtype=torch.float32))
        self.register_buffer('weight', F.softmax(weight, dim=-1))

    def forward(self, probs, target):
        # probs: [B, C] class probabilities
        yq = torch.gather(probs, 1, target.unsqueeze(1))
        lq = (1 - yq ** self.q) / self.q
        weight = torch.gather(self.weight.expand_as(probs), 1, target.unsqueeze(1))
        return (self.alpha * lq + (1 - self.alpha) * lq * weight).mean()

    def set_weights(self, weights):
        weight = torch.log(1 / torch.as_tensor(weights, dtype=torch.float32, device=self.weight.device))
        self.weight.copy_(F.softmax(weight, dim=-1))


def classification_metrics(preds, trues, metric_names):
    """Compute named metrics per head. Returns {metric_name: value}.

    ``metric_names`` comes from ``config['metrics']`` so CASC and LCR report the
    same keys; currently supported: 'acc', 'f1' (macro).
    """
    from sklearn.metrics import accuracy_score, f1_score

    out = {}
    for name in metric_names:
        if name == 'acc':
            out[name] = accuracy_score(trues, preds)
        elif name == 'f1':
            out[name] = f1_score(trues, preds, average='macro', zero_division=0)
        else:
            raise ValueError(f'unknown metric: {name!r}')
    return out


def evaluate(preds, loss_fn, metric_names):
    """Unified evaluation shared by the CASC and LCR train/eval paths.

    ``preds`` is the list of per-batch dicts returned by a model's
    :meth:`~lcr_plus_casc.classifiers.casc.BERTLinear.predict` (keys ``cat``,
    ``pol``, ``cats``, ``pols``, all on CPU). ``loss_fn(out, cats, pols)``
    returns the scalar loss for one batch, where ``out`` is
    ``{'cat': cat_logits, 'pol': pol_logits}``.

    Returns ``{'loss', cat_<metric>, pol_<metric> for each config metric}``.
    """
    cat_logits = [p['cat'] for p in preds]
    pol_logits = [p['pol'] for p in preds]
    total_loss = sum(loss_fn({'cat': lc, 'pol': lp}, p['cats'], p['pols']).item()
                     for lc, lp, p in zip(cat_logits, pol_logits, preds))
    cat_pred = torch.cat(cat_logits).argmax(1).numpy()
    pol_pred = torch.cat(pol_logits).argmax(1).numpy()
    true_cats = torch.cat([p['cats'] for p in preds]).numpy()
    true_pols = torch.cat([p['pols'] for p in preds]).numpy()
    metrics = {'loss': total_loss / max(len(preds), 1)}
    for k, v in classification_metrics(cat_pred, true_cats, metric_names).items():
        metrics[f'cat_{k}'] = v
    for k, v in classification_metrics(pol_pred, true_pols, metric_names).items():
        metrics[f'pol_{k}'] = v
    return metrics


def _weighted_row_mean(per_row, target, weights):
    """Mean of per-row losses, optionally scaled by the true-class weight per row."""
    if weights is None:
        return per_row.mean()
    w = torch.as_tensor(weights, dtype=per_row.dtype, device=per_row.device)
    return (per_row * w[target]).mean()


def _weighted_ce(logits, target, weights):
    per_row = F.cross_entropy(logits, target, reduction='none')
    return _weighted_row_mean(per_row, target, weights)


def _weighted_gce(logits, target, q, weights):
    log_p = F.log_softmax(logits, dim=1)
    p_true = torch.gather(log_p, 1, target.unsqueeze(1)).squeeze(1).exp()
    return _weighted_row_mean((1 - p_true ** q) / q, target, weights)


def absa_loss(out, cats, pols, q=None, l1=0.0, l2=0.0, reg_params=(),
              cat_weights=None, pol_weights=None):
    """Model-agnostic ABSA loss shared by the CASC and LCR trainers.

    ``(CE_cat + CE_pol) / 2`` plus, when ``q > 0``, ``(GCE_q(cat) + GCE_q(pol)) / 2``,
    plus optional L1/L2 penalties over ``reg_params``. ``cat_weights`` /
    ``pol_weights`` (per-class counts) apply to both the CE and GCE terms.
    """
    ce = (_weighted_ce(out['cat'], cats, cat_weights)
          + _weighted_ce(out['pol'], pols, pol_weights)) / 2
    loss = ce
    if q is not None and q > 0:
        loss = loss + (_weighted_gce(out['cat'], cats, q, cat_weights)
                       + _weighted_gce(out['pol'], pols, q, pol_weights)) / 2
    if (l1 or l2) and reg_params:
        reg = torch.zeros((), device=out['cat'].device)
        for p in reg_params:
            if l1:
                reg = reg + l1 * p.abs().sum()
            if l2:
                reg = reg + l2 * (p ** 2).sum()
        loss = loss + reg
    return loss


class GCEQ(nn.Module):
    """Generalized cross entropy with exponent q, applied on the true-class probability."""

    def __init__(self, q):
        super().__init__()
        self.q = q

    def forward(self, logits, y_true):
        # logits: raw logits [B, C]; y_true: class indices [B]
        log_p = F.log_softmax(logits, dim=1)
        p_true = torch.gather(log_p, 1, y_true.unsqueeze(1)).squeeze(1).exp()
        return ((1 - p_true ** self.q) / self.q).mean()
