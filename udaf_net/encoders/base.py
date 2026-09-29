"""Common feature encoder protocol."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn


class FeatureEncoder(nn.Module, ABC):
    feature_dim: int

    @abstractmethod
    def forward_tokens(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return a token sequence shaped ``[batch, tokens, feature_dim]``."""

    def forward_features(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.forward_tokens(inputs).mean(dim=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.forward_features(inputs)

