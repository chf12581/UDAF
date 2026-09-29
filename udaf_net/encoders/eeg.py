"""Reusable EEGNet and convolutional-Transformer baseline encoders."""

from __future__ import annotations

import torch
from torch import nn

from .base import FeatureEncoder


class SeparableConv2d(nn.Module):
    def __init__(self, channels_in: int, channels_out: int, kernel_size: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            channels_in,
            channels_in,
            kernel_size=(1, kernel_size),
            padding="same",
            groups=channels_in,
            bias=False,
        )
        self.pointwise = nn.Conv2d(channels_in, channels_out, kernel_size=1, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(inputs))


class EEGNetEncoder(FeatureEncoder):
    """EEGNet-style encoder preserving temporal tokens for cross-modal fusion."""

    def __init__(
        self,
        n_channels: int = 30,
        dropout: float = 0.5,
        temporal_kernel: int = 64,
        f1: int = 8,
        depth_multiplier: int = 2,
        f2: int = 16,
    ) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.feature_dim = f2
        self.block1 = nn.Sequential(
            nn.Conv2d(
                1,
                f1,
                kernel_size=(1, temporal_kernel),
                padding="same",
                bias=False,
            ),
            nn.BatchNorm2d(f1),
            nn.Conv2d(
                f1,
                f1 * depth_multiplier,
                kernel_size=(n_channels, 1),
                groups=f1,
                bias=False,
            ),
            nn.BatchNorm2d(f1 * depth_multiplier),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(dropout),
        )
        self.block2 = nn.Sequential(
            SeparableConv2d(f1 * depth_multiplier, f2, kernel_size=16),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(dropout),
        )

    @staticmethod
    def _normalise_shape(inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim == 3:
            inputs = inputs.unsqueeze(1)
        if inputs.ndim != 4 or inputs.shape[1] != 1:
            raise ValueError(f"Expected EEG [B,C,T] or [B,1,C,T], got {tuple(inputs.shape)}.")
        return inputs

    def forward_tokens(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.block2(self.block1(self._normalise_shape(inputs)))
        return features.squeeze(2).transpose(1, 2).contiguous()


class EEGConformerLiteEncoder(FeatureEncoder):
    """A self-contained convolutional Transformer baseline.

    It mirrors the convolution-plus-attention intent of the Kaggle EEGConformer route
    without depending on the missing notebook-local ``eegconformer.py`` implementation.
    """

    def __init__(
        self,
        n_channels: int = 30,
        n_samples: int = 500,
        embedding_dim: int = 40,
        depth: int = 4,
        heads: int = 5,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if embedding_dim % heads:
            raise ValueError("embedding_dim must be divisible by heads.")
        self.n_channels = n_channels
        self.feature_dim = embedding_dim
        self.patch_embedding = nn.Sequential(
            nn.Conv2d(1, embedding_dim, kernel_size=(1, 25), padding=(0, 12), bias=False),
            nn.Conv2d(
                embedding_dim,
                embedding_dim,
                kernel_size=(n_channels, 1),
                groups=embedding_dim,
                bias=False,
            ),
            nn.BatchNorm2d(embedding_dim),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4)),
            nn.Dropout(dropout),
        )
        max_tokens = n_samples // 4 + 2
        self.class_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        self.position = nn.Parameter(torch.zeros(1, max_tokens, embedding_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(embedding_dim)
        nn.init.trunc_normal_(self.class_token, std=0.02)
        nn.init.trunc_normal_(self.position, std=0.02)

    def forward_tokens(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim == 3:
            inputs = inputs.unsqueeze(1)
        if inputs.ndim != 4 or inputs.shape[1] != 1:
            raise ValueError(f"Expected EEG [B,C,T] or [B,1,C,T], got {tuple(inputs.shape)}.")
        features = self.patch_embedding(inputs).squeeze(2).transpose(1, 2)
        cls = self.class_token.expand(features.shape[0], -1, -1)
        features = torch.cat((cls, features), dim=1)
        if features.shape[1] > self.position.shape[1]:
            raise ValueError("EEG input is longer than the configured n_samples.")
        return self.norm(self.transformer(features + self.position[:, : features.shape[1]]))

    def forward_features(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.forward_tokens(inputs)[:, 0]

