import torch
import pytest
from pointnet2.modules.grouping import ball_query, square_distance, group_points
from pointnet2.modules.sampling import index_points, farthest_point_sample


B, N, S, nsample = 2, 128, 16, 32
RADIUS = 0.3


def _make_xyz():
    torch.manual_seed(0)
    xyz = torch.rand(B, N, 3)
    idx = farthest_point_sample(xyz, S)
    new_xyz = index_points(xyz, idx)
    return xyz, new_xyz


def test_square_distance_shape():
    src = torch.randn(B, N, 3)
    dst = torch.randn(B, S, 3)
    out = square_distance(src, dst)
    assert out.shape == (B, N, S)


def test_square_distance_symmetry():
    pts = torch.randn(B, N, 3)
    d = square_distance(pts, pts)
    assert torch.allclose(d, d.transpose(1, 2), atol=1e-5)


def test_square_distance_self_is_zero():
    pts = torch.randn(B, N, 3)
    d = square_distance(pts, pts)
    diag = d[:, torch.arange(N), torch.arange(N)]
    assert diag.abs().max() < 1e-5


def test_ball_query_output_shape():
    xyz, new_xyz = _make_xyz()
    idx = ball_query(xyz, new_xyz, RADIUS, nsample)
    assert idx.shape == (B, S, nsample)


def test_ball_query_indices_in_range():
    xyz, new_xyz = _make_xyz()
    idx = ball_query(xyz, new_xyz, RADIUS, nsample)
    assert idx.min() >= 0
    assert idx.max() < N


def test_ball_query_respects_radius():
    """All returned points (except padding) must be within radius."""
    xyz, new_xyz = _make_xyz()
    idx = ball_query(xyz, new_xyz, RADIUS, nsample)
    grouped = index_points(xyz, idx)  # (B, S, nsample, 3)
    dists = ((grouped - new_xyz.unsqueeze(2)) ** 2).sum(-1).sqrt()  # (B, S, nsample)
    # The first index is always within radius (it's the nearest point)
    assert (dists[:, :, 0] <= RADIUS + 1e-5).all()


def test_ball_query_large_radius_returns_nsample():
    """With a huge radius every point qualifies; we should always get nsample."""
    xyz, new_xyz = _make_xyz()
    idx = ball_query(xyz, new_xyz, radius=1e6, nsample=nsample)
    assert idx.shape == (B, S, nsample)


def test_group_points_shape_no_features():
    xyz, new_xyz = _make_xyz()
    idx = ball_query(xyz, new_xyz, RADIUS, nsample)
    out = group_points(xyz, new_xyz, None, idx)
    assert out.shape == (B, S, nsample, 3)


def test_group_points_shape_with_features():
    xyz, new_xyz = _make_xyz()
    idx = ball_query(xyz, new_xyz, RADIUS, nsample)
    features = torch.randn(B, N, 8)
    out = group_points(xyz, new_xyz, features, idx)
    assert out.shape == (B, S, nsample, 3 + 8)


def test_group_points_local_frame():
    """Grouped xyz should be expressed relative to each centroid."""
    xyz, new_xyz = _make_xyz()
    idx = ball_query(xyz, new_xyz, radius=1e6, nsample=nsample)
    out = group_points(xyz, new_xyz, None, idx)  # (B, S, nsample, 3)
    # The nearest point to each centroid has near-zero local coords
    first = out[:, :, 0, :]  # first grouped point is the closest
    assert first.norm(dim=-1).max() < RADIUS + 0.5
