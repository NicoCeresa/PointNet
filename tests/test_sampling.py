import torch
import pytest
from pointnet2.modules.sampling import farthest_point_sample, index_points


B, N, C = 2, 64, 3


def test_fps_output_shape():
    xyz = torch.randn(B, N, C)
    idx = farthest_point_sample(xyz, npoint=16)
    assert idx.shape == (B, 16)


def test_fps_indices_in_range():
    xyz = torch.randn(B, N, C)
    idx = farthest_point_sample(xyz, npoint=16)
    assert idx.min() >= 0
    assert idx.max() < N


def test_fps_no_duplicates_within_sample():
    """FPS should return unique indices for each batch element."""
    torch.manual_seed(0)
    xyz = torch.randn(B, N, C)
    idx = farthest_point_sample(xyz, npoint=16)
    for b in range(B):
        assert len(idx[b].unique()) == 16, "FPS returned duplicate indices"


def test_fps_coverage_better_than_random():
    """FPS sampled points should span a larger volume than random sampling."""
    torch.manual_seed(42)
    xyz = torch.rand(1, 512, 3)
    npoint = 32

    fps_idx = farthest_point_sample(xyz, npoint)
    fps_pts = index_points(xyz, fps_idx)[0]
    fps_span = (fps_pts.max(0)[0] - fps_pts.min(0)[0]).prod()

    rand_idx = torch.randperm(512)[:npoint]
    rand_pts = xyz[0, rand_idx]
    rand_span = (rand_pts.max(0)[0] - rand_pts.min(0)[0]).prod()

    assert fps_span > rand_span, "FPS should cover more space than random sampling"


def test_index_points_shape():
    points = torch.randn(B, N, 8)
    idx = torch.randint(0, N, (B, 16, 4))
    out = index_points(points, idx)
    assert out.shape == (B, 16, 4, 8)
