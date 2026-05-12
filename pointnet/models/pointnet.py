import torch
import torch.nn as nn
import torch.nn.functional as F


class TNet(nn.Module):
    """Mini-PointNet predicting a k×k spatial transformer matrix."""

    def __init__(self, k: int):
        super().__init__()
        self.k = k
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, k, N) → (B, k, k)"""
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.max(dim=2)[0]  # (B, 1024)
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)  # (B, k*k)
        identity = torch.eye(self.k, device=x.device).flatten().unsqueeze(0)
        return (x + identity).view(-1, self.k, self.k)


class PointNetBackbone(nn.Module):
    """
    Shared backbone: steps 1-7 of PointNet.

    Returns:
        local_feat:  (B, N, 64)  — per-point features after feature transform (step 5)
        global_feat: (B, 1024)   — global descriptor after max pooling (step 7)
    """

    def __init__(self):
        super().__init__()
        self.tnet1 = TNet(3)
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.tnet2 = TNet(64)
        self.conv3 = nn.Conv1d(64, 64, 1)
        self.conv4 = nn.Conv1d(64, 128, 1)
        self.conv5 = nn.Conv1d(128, 1024, 1)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(1024)

    def forward(self, xyz: torch.Tensor):
        """xyz: (B, N, 3) → (local_feat, global_feat)"""
        x = xyz.transpose(1, 2)                   # (B, 3, N)
        x = torch.bmm(self.tnet1(x), x)           # input transform → (B, 3, N)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))        # MLP(64, 64) → (B, 64, N)
        x = torch.bmm(self.tnet2(x), x)           # feature transform → (B, 64, N)
        local_feat = x.transpose(1, 2)             # (B, N, 64)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))        # MLP(64, 128, 1024) → (B, 1024, N)
        global_feat = x.max(dim=2)[0]              # (B, 1024)
        return local_feat, global_feat


class PointNetCls(nn.Module):
    """
    Classification model for point clouds, as described in the original PointNet paper.

    Architecture:
    1. input: n x 3 <x,y,z>

       1.a T-Net (input transform) -> 3x3 transform matrix

    2. Mat mul: input @ 3x3 T-net  transform matrix
        -> n x 3 matrix

    3. MLP(64, 64)

    4. T-Net (feature transform) -> 64x64 transform matrix

    5. Mat mul: features @ 64x64 T-net transform matrix
        -> n x 64 matrix

    6. MLP(64, 128, 1024)

    7. Max pooling -> 1024-dim global feature vector

    8. MLP(512, 256, num_classes)

    9. output: num_classes-dim logits
    """

    def __init__(self, num_classes: int = 40):
        super().__init__()
        self.backbone = PointNetBackbone()
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(0.3)

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """
        xyz: (B, N, 3)
        returns: (B, num_classes) logits
        """
        _, global_feat = self.backbone(xyz)
        x = F.relu(self.bn1(self.fc1(global_feat)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        return self.fc3(x)


class PointNetPartSeg(nn.Module):
    """
    Part segmentation model for point clouds.

    Extends the backbone by concatenating per-point local features (step 5),
    the tiled global feature (step 7), and a one-hot category embedding,
    then applying a per-point MLP to produce per-point class logits.

    Segmentation architecture:

    1. n x 1088

    2. MLP(512, 256, 128)

    3. n x 128

    4. MLP(128, num_categories)

    5. output: n x m output logits, where m is the number of part classes for the category
    """

    def __init__(self, num_classes: int = 50, num_categories: int = 16):
        super().__init__()
        self.backbone = PointNetBackbone()
        self.num_categories = num_categories
        in_dim = 64 + 1024 + num_categories
        self.conv1 = nn.Conv1d(in_dim, 512, 1)
        self.conv2 = nn.Conv1d(512, 256, 1)
        self.conv3 = nn.Conv1d(256, 128, 1)
        self.conv4 = nn.Conv1d(128, num_classes, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, xyz: torch.Tensor, category: torch.Tensor) -> torch.Tensor:
        """
        xyz:      (B, N, 3)
        category: (B,) integer category indices
        returns:  (B, N, num_classes) logits
        """
        local_feat, global_feat = self.backbone(xyz)   # (B, N, 64), (B, 1024)
        B, N, _ = local_feat.shape

        cat_onehot = torch.zeros(B, self.num_categories, device=xyz.device)
        cat_onehot.scatter_(1, category.unsqueeze(1), 1)          # (B, num_categories)

        global_tiled = global_feat.unsqueeze(1).expand(-1, N, -1) # (B, N, 1024)
        cat_tiled    = cat_onehot.unsqueeze(1).expand(-1, N, -1)  # (B, N, num_categories)

        x = torch.cat([local_feat, global_tiled, cat_tiled], dim=2).transpose(1, 2)  # (B, 1104, N)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.conv4(x).transpose(1, 2)  # (B, N, num_classes)
