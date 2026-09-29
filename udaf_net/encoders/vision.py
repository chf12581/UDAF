"""Visual encoders corresponding to the Kaggle ResNet18 and ViT baselines."""

from __future__ import annotations

import torch
from torch import nn

from .base import FeatureEncoder


class ResNet18Encoder(FeatureEncoder):
    def __init__(self, pretrained: bool = False) -> None:
        super().__init__()
        try:
            from torchvision.models import ResNet18_Weights, resnet18
        except ImportError as exc:
            raise ImportError("ResNet18Encoder requires torchvision.") from exc

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)
        self.feature_dim = int(backbone.fc.in_features)
        self.features = nn.Sequential(*list(backbone.children())[:-2])

    def forward_tokens(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.features(inputs)
        return features.flatten(2).transpose(1, 2).contiguous()


class TorchvisionViTEncoder(FeatureEncoder):
    """TorchVision ViT-B/16 encoder available in the ``eegpy310`` environment."""

    def __init__(self, pretrained: bool = False) -> None:
        super().__init__()
        try:
            from torchvision.models import ViT_B_16_Weights, vit_b_16
        except ImportError as exc:
            raise ImportError("TorchvisionViTEncoder requires torchvision.") from exc
        weights = ViT_B_16_Weights.DEFAULT if pretrained else None
        self.model = vit_b_16(weights=weights)
        self.feature_dim = int(self.model.hidden_dim)

    def forward_tokens(self, inputs: torch.Tensor) -> torch.Tensor:
        tokens = self.model._process_input(inputs)
        class_token = self.model.class_token.expand(tokens.shape[0], -1, -1)
        return self.model.encoder(torch.cat((class_token, tokens), dim=1))

    def forward_features(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.forward_tokens(inputs)[:, 0]


class HuggingFaceViTEncoder(FeatureEncoder):
    def __init__(self, model_name_or_path: str, *, local_files_only: bool = False) -> None:
        super().__init__()
        try:
            from transformers import AutoModel
        except ImportError as exc:
            raise ImportError(
                "ViT baselines require the optional dependency: pip install -e '.[transformer]'"
            ) from exc
        self.model = AutoModel.from_pretrained(
            model_name_or_path,
            local_files_only=local_files_only,
        )
        self.feature_dim = int(self.model.config.hidden_size)

    def forward_tokens(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(pixel_values=inputs).last_hidden_state

    def forward_features(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.forward_tokens(inputs)[:, 0]
