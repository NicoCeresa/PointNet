import torch
import torch.nn as nn


class PointNetPPCls(nn.Module):
    def __init__(self, num_classes: int = 40, use_msg: bool = False):
        super().__init__()
        raise NotImplementedError

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """
        xyz: (B, N, 3)
        returns: (B, num_classes) logits
        """
        raise NotImplementedError


class PointNetPPPartSeg(nn.Module):
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


class PointNetPPSemSeg(nn.Module):
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
