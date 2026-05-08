import torch
import pytest
from pointnet2.modules import (
    PointNetSetAbstraction,
    PointNetSetAbstractionGlobal,
    PointNetSetAbstractionMsg,
    PointNetFeaturePropagation,
)

B, N = 2, 256


def test_sa_ssg_no_features():
    sa = PointNetSetAbstraction(npoint=64, radius=0.2, nsample=16, in_channel=0, mlp=[32, 64])
    xyz = torch.rand(B, N, 3)
    new_xyz, new_pts = sa(xyz, None)
    assert new_xyz.shape == (B, 64, 3)
    assert new_pts.shape == (B, 64, 64)


def test_sa_ssg_with_features():
    sa = PointNetSetAbstraction(npoint=64, radius=0.3, nsample=16, in_channel=8, mlp=[32, 64])
    xyz = torch.rand(B, N, 3)
    pts = torch.rand(B, N, 8)
    new_xyz, new_pts = sa(xyz, pts)
    assert new_xyz.shape == (B, 64, 3)
    assert new_pts.shape == (B, 64, 64)


def test_sa_msg_output_channels():
    """MSG concatenates outputs from all scales."""
    sa = PointNetSetAbstractionMsg(
        npoint=64,
        radius_list=[0.1, 0.2],
        nsample_list=[16, 32],
        in_channel=0,
        mlp_list=[[32, 64], [64, 128]],
    )
    xyz = torch.rand(B, N, 3)
    new_xyz, new_pts = sa(xyz, None)
    assert new_xyz.shape == (B, 64, 3)
    assert new_pts.shape == (B, 64, 64 + 128)


def test_sa_global_output_shape():
    sa = PointNetSetAbstractionGlobal(in_channel=64, mlp=[128, 256])
    xyz = torch.rand(B, N, 3)
    pts = torch.rand(B, N, 64)
    none_xyz, global_feat = sa(xyz, pts)
    assert none_xyz is None
    assert global_feat.shape == (B, 256)


def test_fp_output_shape():
    fp = PointNetFeaturePropagation(in_channel=64 + 32, mlp=[64, 32])
    xyz1 = torch.rand(B, N, 3)         # fine (upsample target)
    xyz2 = torch.rand(B, 32, 3)        # coarse
    pts1 = torch.rand(B, N, 64)        # skip features
    pts2 = torch.rand(B, 32, 32)       # coarse features
    out = fp(xyz1, xyz2, pts1, pts2)
    assert out.shape == (B, N, 32)


def test_fp_no_skip():
    fp = PointNetFeaturePropagation(in_channel=32, mlp=[64])
    xyz1 = torch.rand(B, N, 3)
    xyz2 = torch.rand(B, 16, 3)
    pts2 = torch.rand(B, 16, 32)
    out = fp(xyz1, xyz2, None, pts2)
    assert out.shape == (B, N, 64)


def test_fp_single_coarse_point():
    """When xyz2 has a single point, features should broadcast to all N."""
    fp = PointNetFeaturePropagation(in_channel=16, mlp=[32])
    xyz1 = torch.rand(B, N, 3)
    xyz2 = torch.rand(B, 1, 3)
    pts2 = torch.rand(B, 1, 16)
    out = fp(xyz1, xyz2, None, pts2)
    assert out.shape == (B, N, 32)


def test_cls_forward_shape():
    from pointnet2.models import PointNet2Cls
    model = PointNet2Cls(num_classes=40, use_msg=False)
    xyz = torch.rand(2, 1024, 3)
    logits = model(xyz)
    assert logits.shape == (2, 40)


def test_semseg_forward_shape():
    from pointnet2.models import PointNet2SemSeg
    model = PointNet2SemSeg(num_classes=13, in_feature_dim=6)
    xyz = torch.rand(2, 512, 3)
    feat = torch.rand(2, 512, 6)
    logits = model(xyz, feat)
    assert logits.shape == (2, 512, 13)
