import torch
from .sampling import index_points


def square_distance(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """
    src: (B, N, C)
    dst: (B, M, C)
    returns: (B, N, M) pairwise squared Euclidean distances
    """
    # torch.cdist computes L2; squaring avoids the sqrt for comparison-only uses
    return torch.cdist(src, dst).pow(2)


def ball_query(
    xyz: torch.Tensor,
    new_xyz: torch.Tensor,
    radius: float,
    nsample: int,
) -> torch.Tensor:
    """
    xyz:     (B, N, 3) all points
    new_xyz: (B, S, 3) centroid (query) points
    returns: (B, S, nsample) indices into xyz

    Returns the nsample closest points within radius. Points outside the ball are
    replaced by the nearest valid neighbor so downstream code never sees padding indices.
    """
    sqrdists = square_distance(new_xyz, xyz)  # (B, S, N)
    # topk with largest=False gives the nsample nearest points
    nearest_dists, idx = sqrdists.topk(nsample, dim=-1, largest=False)  # (B, S, nsample)

    # Replace out-of-radius points with the nearest valid neighbor (idx[:,:,0])
    first = idx[:, :, 0:1].expand_as(idx)
    mask = nearest_dists > radius ** 2
    idx[mask] = first[mask]

    return idx


def knn_query(xyz: torch.Tensor, new_xyz: torch.Tensor, k: int) -> torch.Tensor:
    """
    xyz:     (B, N, 3)
    new_xyz: (B, S, 3)
    returns: (B, S, k) indices of k nearest neighbors in xyz for each point in new_xyz
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, idx = sqrdists.topk(k, dim=-1, largest=False)
    return idx


def group_points(
    xyz: torch.Tensor,
    new_xyz: torch.Tensor,
    points: torch.Tensor | None,
    idx: torch.Tensor,
) -> torch.Tensor:
    """
    Gather and locally normalize grouped points.

    xyz:    (B, N, 3)
    new_xyz:(B, S, 3)
    points: (B, N, C) or None
    idx:    (B, S, nsample)
    returns:(B, S, nsample, 3+C)
    """
    grouped_xyz = index_points(xyz, idx)           # (B, S, nsample, 3)
    grouped_xyz = grouped_xyz - new_xyz.unsqueeze(2)  # local frame

    if points is not None:
        grouped_feat = index_points(points, idx)   # (B, S, nsample, C)
        return torch.cat([grouped_xyz, grouped_feat], dim=-1)
    return grouped_xyz
