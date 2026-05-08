import torch


def square_distance(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """
    src: (B, N, C)
    dst: (B, M, C)
    returns: (B, N, M) pairwise squared Euclidean distances
    """
    raise NotImplementedError


def ball_query(
    xyz: torch.Tensor,
    new_xyz: torch.Tensor,
    radius: float,
    nsample: int,
) -> torch.Tensor:
    """
    xyz:     (B, N, 3) all points
    new_xyz: (B, S, 3) centroid points
    returns: (B, S, nsample) indices into xyz
    """
    raise NotImplementedError


def group_points(
    xyz: torch.Tensor,
    new_xyz: torch.Tensor,
    points: torch.Tensor | None,
    idx: torch.Tensor,
) -> torch.Tensor:
    """
    Gather grouped points and express xyz relative to each centroid.

    xyz:    (B, N, 3)
    new_xyz:(B, S, 3)
    points: (B, N, C) or None
    idx:    (B, S, nsample)
    returns:(B, S, nsample, 3+C)
    """
    raise NotImplementedError
