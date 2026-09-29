"""Equations (1)--(3), (8)--(11) of the supplied manuscript.

The fusion head accepts encoder tokens. Encoder training is in train.py.
This is a manuscript-aligned assembly, NOT an alias for a historical checkpoint.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class Posterior(nn.Module):
    """Eq. (2)--(3). Variance input is detached; v0 is variance, not std."""
    def __init__(self, dimension=256, hidden=128, v0=0.01, eps=1e-6):
        super().__init__()
        if v0 <= 0 or eps <= 0:
            raise ValueError("v0 and epsilon must be positive")
        self.shared = nn.Sequential(nn.LayerNorm(dimension), nn.Linear(dimension, hidden), nn.GELU())
        self.mean = nn.Linear(hidden, dimension)
        self.variance = nn.Linear(hidden, dimension)
        for head in (self.mean, self.variance):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        self.register_buffer("v0", torch.full((dimension,), v0))
        self.eps = eps

    def forward(self, z):
        h = self.shared(z)
        mu = z + self.mean(h)
        v = self.v0 * (self.eps + F.softplus(self.variance(h.detach()))) / (self.eps + math.log(2))
        return mu, v


def antithetic(mu, variance, count, generator=None):
    """Eq. (8): mean first, then alternating +epsilon and -epsilon."""
    if not isinstance(count, int) or count < 1 or count % 2 != 1:
        raise ValueError("Sample count must be a positive odd integer")
    if count == 1:
        return mu[:, None]
    noise = torch.randn((len(mu), count // 2, *mu.shape[1:]),
                        dtype=mu.dtype, device=mu.device, generator=generator)
    delta = variance.sqrt()[:, None] * noise
    pairs = torch.stack((mu[:, None] + delta, mu[:, None] - delta), dim=2)
    return torch.cat((mu[:, None], pairs.flatten(1, 2)), dim=1)


class ReliabilityGuidedFusion(nn.Module):
    """Eq. (9)--(10). Shared content/variance compression, separate excitations."""
    def __init__(self, dimension=256, hidden=128, eps=1e-6):
        super().__init__()
        self.content = nn.Linear(2 * dimension, hidden)
        self.uncertainty = nn.Linear(2 * dimension, hidden)
        nn.init.zeros_(self.uncertainty.weight)
        nn.init.zeros_(self.uncertainty.bias)
        self.eeg_gate = nn.Linear(hidden, dimension)
        self.face_gate = nn.Linear(hidden, dimension)
        self.eps = eps

    def forward(self, ze, zf, ve=None, vf=None, v0e=None, v0f=None):
        content = torch.cat((ze.mean(2), zf.mean(2)), dim=-1)
        uncertainty = torch.zeros_like(content[:, 0])
        if ve is not None:
            uncertainty = torch.cat((((ve + self.eps) / (v0e + self.eps)).log().mean(1),
                                     ((vf + self.eps) / (v0f + self.eps)).log().mean(1)), dim=-1)
        shared = F.relu(self.content(content) + self.uncertainty(uncertainty)[:, None])
        ge, gf = self.eeg_gate(shared).sigmoid(), self.face_gate(shared).sigmoid()
        return ze * ge[:, :, None], zf * gf[:, :, None], ge, gf


class UDAFNet(nn.Module):
    """Four manuscript configurations: full, base, uda, rgf.

    base/uda use bidirectional cross-attention; rgf/full use channel gates.
    Hidden sizes, attention details and dropout are configurable implementation
    choices where the manuscript does not specify a unique value.
    """
    def __init__(self, eeg_dimension=16, face_dimension=512, dimension=256,
                 hidden=128, num_classes=5, dropout=0.3, v0=0.01,
                 eps=1e-6, variant="full", attention_heads=4):
        super().__init__()
        if variant not in ("full", "base", "uda", "rgf"):
            raise ValueError("Unknown variant")
        self.variant, self.num_classes, self.eps = variant, num_classes, eps
        self.uses_uda = variant in ("full", "uda")
        self.eeg_projection = nn.Linear(eeg_dimension, dimension)
        self.face_projection = nn.Linear(face_dimension, dimension)
        if self.uses_uda:
            self.eeg_posterior = Posterior(dimension, hidden, v0, eps)
            self.face_posterior = Posterior(dimension, hidden, v0, eps)
            # Fixed score calibration defaults; manuscript gives no numerical values.
            self.register_buffer("mls_scale", torch.tensor(1.0))
            self.register_buffer("mls_shift", torch.tensor(0.0))
        if variant in ("full", "rgf"):
            self.rgf = ReliabilityGuidedFusion(dimension, hidden, eps)
            if variant == "rgf":
                self.rgf.uncertainty.requires_grad_(False)
        else:
            self.eeg_attention = nn.MultiheadAttention(dimension, attention_heads, dropout=dropout, batch_first=True)
            self.face_attention = nn.MultiheadAttention(dimension, attention_heads, dropout=dropout, batch_first=True)
            self.eeg_norm, self.face_norm = nn.LayerNorm(dimension), nn.LayerNorm(dimension)
        self.classifier = nn.Sequential(nn.LayerNorm(2 * dimension), nn.Dropout(dropout),
                                        nn.Linear(2 * dimension, num_classes))

    def forward(self, eeg_tokens, face_tokens, samples=3, generator=None):
        e, f = self.eeg_projection(eeg_tokens), self.face_projection(face_tokens)
        out = {"base_eeg": e, "base_face": f}
        if self.uses_uda:
            me, ve = self.eeg_posterior(e)
            mf, vf = self.face_posterior(f)
            ze, zf = antithetic(me, ve, samples, generator), antithetic(mf, vf, samples, generator)
            out.update(mu_eeg=me, mu_face=mf, var_eeg=ve, var_face=vf)
        else:
            ze, zf = e[:, None], f[:, None]
        if self.variant in ("full", "rgf"):
            args = (ve, vf, self.eeg_posterior.v0, self.face_posterior.v0) if self.uses_uda else ()
            ze, zf, ge, gf = self.rgf(ze, zf, *args)
            out.update(gate_eeg=ge, gate_face=gf)
        else:
            b, k = ze.shape[:2]
            eflat, fflat = ze.flatten(0, 1), zf.flatten(0, 1)
            ae = self.eeg_norm(eflat + self.eeg_attention(eflat, fflat, fflat, need_weights=False)[0])
            af = self.face_norm(fflat + self.face_attention(fflat, eflat, eflat, need_weights=False)[0])
            ze, zf = ae.reshape(b, k, *ae.shape[1:]), af.reshape(b, k, *af.shape[1:])
        logits = self.classifier(torch.cat((ze.mean(2), zf.mean(2)), dim=-1))
        # Eq. (11): mean of probabilities, never softmax(mean(logits)).
        logp = torch.logsumexp(logits.log_softmax(-1), dim=1) - math.log(logits.shape[1])
        out.update(log_probabilities=logp, probabilities=logp.exp(), sample_logits=logits)
        return out
