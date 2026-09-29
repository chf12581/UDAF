"""Equations (4)--(7), (12). Explicitly preserve the nested loss weights."""
import torch
from torch.nn import functional as F


def trial_mean(values, trials):
    _, inverse, counts = trials.unique(return_inverse=True, return_counts=True)
    sums = values.new_zeros((len(counts), *values.shape[1:])).index_add(0, inverse, values)
    means = sums / counts.reshape((-1,) + (1,) * (values.ndim - 1))
    return means, inverse


def balanced_mean(values, trials):
    return trial_mean(values, trials)[0].mean()


def relation_loss(scores, labels, trials):
    """Eq. (7), E->F rows; no same-class/different-trial positives.

    Positive/negative masks each have unit mass per anchor. Missing negative
    relations contribute zero (no NaN). Equal q recovers the printed 1/(2B).
    """
    pos = trials[:, None].eq(trials[None, :])
    neg = labels[:, None].ne(labels[None, :])
    if bool((pos & neg).any()):
        raise ValueError("One trial cannot have conflicting class labels")
    wp = pos.to(scores.dtype) / pos.sum(1, keepdim=True).clamp_min(1)
    wn = neg.to(scores.dtype) / neg.sum(1, keepdim=True).clamp_min(1)
    return 0.5 * balanced_mean((wp * F.softplus(-scores) + wn * F.softplus(scores)).sum(1), trials)


def objectives(out, labels, trials, model, weights=None):
    w = {"det": 0.05, "adapter": 0.1, "uncertainty_ratio": 0.1, "mls": 0.05, "cross": 1e-4}
    if weights:
        w.update(weights)
    ce = balanced_mean(F.nll_loss(out["log_probabilities"], labels, reduction="none"), trials)
    result = {"classification": ce}
    if not model.uses_uda:
        return {**result, "total": ce}
    e, f = out["base_eeg"], out["base_face"]
    te, inv = trial_mean(e.mean(1).detach(), trials)
    tf, _ = trial_mean(f.mean(1).detach(), trials)
    te, tf = te[inv], tf[inv]
    me, mf = out["mu_eeg"].mean(1), out["mu_face"].mean(1)
    ve, vf = out["var_eeg"].mean(1), out["var_face"].mean(1)
    def mean_error(mu, target):
        return balanced_mean((mu - target).abs().mean(-1), trials)
    def uncertainty(mu, v, target):
        value = 0.5 * ((v + model.eps).log() + (target - mu.detach()).square() / (v + model.eps)).mean(-1)
        return balanced_mean(value, trials)
    mean_self = 0.5 * (mean_error(me, te) + mean_error(mf, tf))
    mean_cross = 0.5 * (mean_error(me, tf) + mean_error(mf, te))
    unc_self = 0.5 * (uncertainty(me, ve, te) + uncertainty(mf, vf, tf))
    unc_cross = 0.5 * (uncertainty(me, ve, tf) + uncertainty(mf, vf, te))
    l_mean, l_unc = mean_self + w["cross"] * mean_cross, unc_self + w["cross"] * unc_cross
    vem = ve + (out["mu_eeg"] - me[:, None]).square().mean(1).detach()
    vfm = vf + (out["mu_face"] - mf[:, None]).square().mean(1).detach()
    var_sum = vem[:, None] + vfm[None] + model.eps
    raw = -0.5 * ((me[:, None] - mf[None]).square() / var_sum + var_sum.log()).mean(-1)
    l_mls = relation_loss(model.mls_scale * raw + model.mls_shift, labels, trials)
    l_det = relation_loss(F.normalize(me, dim=-1) @ F.normalize(mf, dim=-1).T, labels, trials)
    total = ce + w["det"] * l_det + w["adapter"] * (l_mean + w["uncertainty_ratio"] * l_unc) + w["mls"] * l_mls
    return {**result, "mean": l_mean, "uncertainty": l_unc, "deterministic": l_det, "mls": l_mls, "total": total}
