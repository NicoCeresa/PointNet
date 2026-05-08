import torch
import torch.nn as nn


class PointNet2SemSeg(nn.Module):
    def __init__(self, num_classes: int = 13, in_feature_dim: int = 6):
        super().__init__()
        raise NotImplementedError

    def forward(self, xyz: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """
        xyz:      (B, N, 3)
        features: (B, N, in_feature_dim)
        returns:  (B, N, num_classes) logits
        """
        raise NotImplementedError
