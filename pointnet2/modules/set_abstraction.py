import torch
import torch.nn as nn


class PointNetSetAbstraction(nn.Module):
    """Single-scale grouping (SSG) set abstraction layer."""

    def __init__(self, npoint: int, radius: float, nsample: int, in_channel: int, mlp: list[int]):
        super().__init__()
        raise NotImplementedError

    def forward(self, xyz: torch.Tensor, points: torch.Tensor | None):
        """
        xyz:    (B, N, 3)
        points: (B, N, C) or None
        returns: new_xyz (B, npoint, 3), new_points (B, npoint, mlp[-1])
        """
        raise NotImplementedError


class PointNetSetAbstractionMsg(nn.Module):
    """Multi-scale grouping (MSG) set abstraction layer."""

    def __init__(
        self,
        npoint: int,
        radius_list: list[float],
        nsample_list: list[int],
        in_channel: int,
        mlp_list: list[list[int]],
    ):
        super().__init__()
        raise NotImplementedError

    def forward(self, xyz: torch.Tensor, points: torch.Tensor | None):
        raise NotImplementedError


class PointNetSetAbstractionGlobal(nn.Module):
    """Global set abstraction — pools all points into a single feature vector."""

    def __init__(self, in_channel: int, mlp: list[int]):
        super().__init__()
        raise NotImplementedError

    def forward(self, xyz: torch.Tensor, points: torch.Tensor | None):
        """returns: None, (B, mlp[-1])"""
        raise NotImplementedError


class PointNetFeaturePropagation(nn.Module):
    """Inverse-distance-weighted interpolation + skip connection + MLP."""

    def __init__(self, in_channel: int, mlp: list[int]):
        super().__init__()
        raise NotImplementedError

    def forward(
        self,
        xyz1: torch.Tensor,
        xyz2: torch.Tensor,
        points1: torch.Tensor | None,
        points2: torch.Tensor,
    ) -> torch.Tensor:
        """
        xyz1:    (B, N, 3) finer resolution (upsample target)
        xyz2:    (B, S, 3) coarser resolution
        points1: (B, N, C1) skip-connection features (or None)
        points2: (B, S, C2) features to upsample
        returns: (B, N, mlp[-1])
        """
        raise NotImplementedError
