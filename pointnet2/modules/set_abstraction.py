import torch
import torch.nn as nn

from .grouping import ball_query, group_points, square_distance
from .sampling import farthest_point_sample, index_points


def _build_mlp_2d(in_ch: int, channels: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for out_ch in channels:
        layers += [nn.Conv2d(in_ch, out_ch, 1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)]
        in_ch = out_ch
    return nn.Sequential(*layers)


def _build_mlp_1d(in_ch: int, channels: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for out_ch in channels:
        layers += [nn.Conv1d(in_ch, out_ch, 1), nn.BatchNorm1d(out_ch), nn.ReLU(inplace=True)]
        in_ch = out_ch
    return nn.Sequential(*layers)


class PointNetSetAbstraction(nn.Module):
    """Single-scale grouping (SSG) set abstraction layer."""

    def __init__(
        self,
        npoint: int,
        radius: float,
        nsample: int,
        in_channel: int,
        mlp: list[int],
    ):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        # in_channel = feature channels (xyz not included); MLP sees xyz + features
        self.mlp = _build_mlp_2d(in_channel + 3, mlp)

    def forward(
        self, xyz: torch.Tensor, points: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        xyz:    (B, N, 3)
        points: (B, N, C) or None
        returns: new_xyz (B, npoint, 3), new_points (B, npoint, mlp[-1])
        """
        idx = farthest_point_sample(xyz, self.npoint)
        new_xyz = index_points(xyz, idx)

        group_idx = ball_query(xyz, new_xyz, self.radius, self.nsample)
        grouped = group_points(xyz, new_xyz, points, group_idx)  # (B, npoint, nsample, 3+C)

        grouped = grouped.permute(0, 3, 1, 2)   # (B, 3+C, npoint, nsample)
        grouped = self.mlp(grouped)              # (B, mlp[-1], npoint, nsample)
        new_points = grouped.max(dim=-1)[0]      # (B, mlp[-1], npoint)
        return new_xyz, new_points.transpose(1, 2)  # (B, npoint, mlp[-1])


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
        assert len(radius_list) == len(nsample_list) == len(mlp_list)
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.mlp_blocks = nn.ModuleList(
            [_build_mlp_2d(in_channel + 3, mlp) for mlp in mlp_list]
        )

    def forward(
        self, xyz: torch.Tensor, points: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        idx = farthest_point_sample(xyz, self.npoint)
        new_xyz = index_points(xyz, idx)

        scale_features = []
        for radius, nsample, mlp in zip(self.radius_list, self.nsample_list, self.mlp_blocks):
            group_idx = ball_query(xyz, new_xyz, radius, nsample)
            grouped = group_points(xyz, new_xyz, points, group_idx)  # (B, npoint, nsample, C)
            grouped = grouped.permute(0, 3, 1, 2)
            grouped = mlp(grouped)
            scale_features.append(grouped.max(dim=-1)[0].transpose(1, 2))  # (B, npoint, C_out)

        new_points = torch.cat(scale_features, dim=-1)
        return new_xyz, new_points


class PointNetSetAbstractionGlobal(nn.Module):
    """Global set abstraction — pools all points into a single feature vector."""

    def __init__(self, in_channel: int, mlp: list[int]):
        super().__init__()
        # in_channel = feature channels; xyz always concatenated
        self.mlp = _build_mlp_1d(in_channel + 3, mlp)

    def forward(
        self, xyz: torch.Tensor, points: torch.Tensor | None
    ) -> tuple[None, torch.Tensor]:
        """
        returns: None, (B, mlp[-1])
        """
        if points is not None:
            x = torch.cat([xyz, points], dim=-1)  # (B, N, 3+C)
        else:
            x = xyz
        x = x.transpose(1, 2)   # (B, 3+C, N)
        x = self.mlp(x)          # (B, mlp[-1], N)
        x = x.max(dim=-1)[0]     # (B, mlp[-1])
        return None, x


class PointNetFeaturePropagation(nn.Module):
    """Inverse-distance-weighted interpolation + skip connection + MLP."""

    def __init__(self, in_channel: int, mlp: list[int]):
        super().__init__()
        # in_channel = C_skip + C_interpolated
        self.mlp = _build_mlp_1d(in_channel, mlp)

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
        B, N, _ = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated = points2.expand(B, N, -1)
        else:
            dists = square_distance(xyz1, xyz2)            # (B, N, S)
            dists, idx = dists.topk(3, dim=-1, largest=False)  # 3-NN
            dists = dists.clamp(min=1e-10)
            weight = 1.0 / dists
            weight = weight / weight.sum(dim=-1, keepdim=True)
            neighbors = index_points(points2, idx)         # (B, N, 3, C2)
            interpolated = (neighbors * weight.unsqueeze(-1)).sum(dim=2)  # (B, N, C2)

        if points1 is not None:
            x = torch.cat([points1, interpolated], dim=-1)
        else:
            x = interpolated

        x = x.transpose(1, 2)   # (B, C, N)
        x = self.mlp(x)
        return x.transpose(1, 2)  # (B, N, mlp[-1])
