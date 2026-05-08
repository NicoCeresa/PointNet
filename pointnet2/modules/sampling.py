import torch


def farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """
    xyz: (B, N, 3)
    returns: (B, npoint) indices of sampled points
    """
    B, N, _ = xyz.shape
    device = xyz.device
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.full((B, N), 1e10, device=device)
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_idx = torch.arange(B, dtype=torch.long, device=device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_idx, farthest, :].unsqueeze(1)  # (B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)
        distance = torch.minimum(distance, dist)
        farthest = distance.argmax(dim=-1)

    return centroids


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    points: (B, N, C)
    idx:    (B, ...) arbitrary index shape
    returns: (B, ..., C)
    """
    B = points.shape[0]
    view_shape = [B] + [1] * (idx.dim() - 1)
    repeat_shape = [1] + list(idx.shape[1:])
    batch_idx = torch.arange(B, device=points.device).view(*view_shape).repeat(*repeat_shape)
    return points[batch_idx, idx, :]
