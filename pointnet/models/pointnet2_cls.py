import torch
import torch.nn as nn


class PointNet2Cls(nn.Module):
    def __init__(self, num_classes: int = 40, use_msg: bool = False):
        super().__init__()
        raise NotImplementedError

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """
        xyz: (B, N, 3)
        returns: (B, num_classes) logits
        """
        raise NotImplementedError
