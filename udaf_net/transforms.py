"""Small tensor-only image transform to avoid eager preprocessing of all video frames."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as functional


class FaceTransform:
    def __init__(self, size: int = 224, *, augment: bool = False) -> None:
        self.size = size
        self.augment = augment
        self.mean = torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
        self.std = torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1)

    def __call__(self, frame: np.ndarray) -> torch.Tensor:
        array = np.asarray(frame)
        if array.ndim != 3:
            raise ValueError(f"Expected HWC image, got {array.shape}.")
        tensor = torch.as_tensor(np.ascontiguousarray(array)).permute(2, 0, 1).float()
        if tensor.shape[0] == 1:
            tensor = tensor.repeat(3, 1, 1)
        if tensor.shape[0] != 3:
            raise ValueError(f"Expected 3 image channels, got {tensor.shape[0]}.")
        if tensor.max().item() > 1.0:
            tensor = tensor / 255.0
        # Prepared DEAP/EAV caches commonly already store the requested
        # resolution.  ``interpolate(x, size=x.shape[-2:])`` is bitwise an
        # identity but incurs a large CPU cost for every frame, so bypass it.
        if tuple(tensor.shape[-2:]) != (self.size, self.size):
            tensor = functional.interpolate(
                tensor.unsqueeze(0),
                size=(self.size, self.size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        if self.augment and torch.rand(()) < 0.5:
            tensor = tensor.flip(-1)
        return (tensor - self.mean) / self.std
