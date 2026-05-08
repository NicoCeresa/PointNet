import torch


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
