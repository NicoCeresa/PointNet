import torch
import torch.nn as nn


class PointNet2PartSeg(nn.Module):
    def __init__(self, num_classes: int = 50, num_categories: int = 16):
        super().__init__()
        raise NotImplementedError

    def forward(self, xyz: torch.Tensor, category: torch.Tensor) -> torch.Tensor:
        """
        xyz:      (B, N, 3)
        category: (B,) integer category indices
        returns:  (B, N, num_classes) logits
        """
        raise NotImplementedError
