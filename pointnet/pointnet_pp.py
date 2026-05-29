import torch
import torch.nn as nn


# ── modules ───────────────────────────────────────────────────────────────────

def square_distance(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """
    src: (B, N, C)
    dst: (B, M, C)
    returns: (B, N, M) pairwise squared Euclidean distances
    """
    raise NotImplementedError


def ball_query(xyz: torch.Tensor, new_xyz: torch.Tensor, radius: float, nsample: int) -> torch.Tensor:
    """
    xyz:     (B, N, 3) all points
    new_xyz: (B, S, 3) centroid points
    returns: (B, S, nsample) indices into xyz
    """
    raise NotImplementedError


def group_points(xyz: torch.Tensor, new_xyz: torch.Tensor, points: torch.Tensor | None, idx: torch.Tensor) -> torch.Tensor:
    """
    Gather grouped points and express xyz relative to each centroid.

    xyz:     (B, N, 3)
    new_xyz: (B, S, 3)
    points:  (B, N, C) or None
    idx:     (B, S, nsample)
    returns: (B, S, nsample, 3+C)
    """
    raise NotImplementedError


def farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """
    xyz: (B, N, 3)
    returns: (B, npoint) indices of sampled points
    """
    raise NotImplementedError


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    points: (B, N, C)
    idx:    (B, ...) arbitrary index shape
    returns: (B, ..., C)
    """
    raise NotImplementedError


# ── set abstraction layers ────────────────────────────────────────────────────

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

    def __init__(self, npoint: int, radius_list: list[float], nsample_list: list[int], in_channel: int, mlp_list: list[list[int]]):
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

    def forward(self, xyz1: torch.Tensor, xyz2: torch.Tensor, points1: torch.Tensor | None, points2: torch.Tensor) -> torch.Tensor:
        """
        xyz1:    (B, N, 3) finer resolution (upsample target)
        xyz2:    (B, S, 3) coarser resolution
        points1: (B, N, C1) skip-connection features (or None)
        points2: (B, S, C2) features to upsample
        returns: (B, N, mlp[-1])
        """
        raise NotImplementedError


# ── models ────────────────────────────────────────────────────────────────────

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
