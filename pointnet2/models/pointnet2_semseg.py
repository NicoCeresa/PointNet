import torch
import torch.nn as nn
import pytorch_lightning as pl
from omegaconf import DictConfig

from pointnet2.modules import (
    PointNetFeaturePropagation,
    PointNetSetAbstraction,
)
from pointnet2.utils.metrics import mean_iou


class PointNet2SemSeg(nn.Module):
    """
    PointNet++ semantic segmentation network for S3DIS.
    Input per point: (x, y, z, r, g, b, nx, ny, nz) = 9 dims.
    xyz channels = 3, feature channels = 6 (the remaining dims).
    """

    def __init__(self, num_classes: int = 13, in_feature_dim: int = 6):
        super().__init__()
        self.sa1 = PointNetSetAbstraction(1024, 0.1, 32, in_channel=in_feature_dim, mlp=[32, 32, 64])
        self.sa2 = PointNetSetAbstraction(256, 0.2, 32, in_channel=64, mlp=[64, 64, 128])
        self.sa3 = PointNetSetAbstraction(64, 0.4, 32, in_channel=128, mlp=[128, 128, 256])
        self.sa4 = PointNetSetAbstraction(16, 0.8, 32, in_channel=256, mlp=[256, 256, 512])

        self.fp4 = PointNetFeaturePropagation(in_channel=512 + 256, mlp=[256, 256])
        self.fp3 = PointNetFeaturePropagation(in_channel=256 + 128, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=256 + 64, mlp=[256, 128])
        self.fp1 = PointNetFeaturePropagation(in_channel=128 + in_feature_dim, mlp=[128, 128, 128])

        self.head = nn.Sequential(
            nn.Conv1d(128, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv1d(128, num_classes, 1),
        )

    def forward(self, xyz: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """
        xyz:      (B, N, 3)
        features: (B, N, in_feature_dim)
        returns:  (B, N, num_classes) logits
        """
        xyz1, f1 = self.sa1(xyz, features)
        xyz2, f2 = self.sa2(xyz1, f1)
        xyz3, f3 = self.sa3(xyz2, f2)
        xyz4, f4 = self.sa4(xyz3, f3)

        f3 = self.fp4(xyz3, xyz4, f3, f4)
        f2 = self.fp3(xyz2, xyz3, f2, f3)
        f1 = self.fp2(xyz1, xyz2, f1, f2)
        f0 = self.fp1(xyz, xyz1, features, f1)

        out = f0.transpose(1, 2)   # (B, 128, N)
        return self.head(out).transpose(1, 2)  # (B, N, num_classes)


class PointNet2SemSegModule(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.model = PointNet2SemSeg(
            num_classes=cfg.model.num_classes,
            in_feature_dim=cfg.model.in_feature_dim,
        )
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, xyz, features):
        return self.model(xyz, features)

    def _step(self, batch, stage: str):
        xyz, features, labels = batch
        logits = self(xyz, features)
        loss = self.criterion(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
        preds = logits.argmax(dim=-1)
        miou = mean_iou(preds, labels, self.cfg.model.num_classes)
        self.log(f"{stage}/loss", loss, prog_bar=True, sync_dist=True)
        self.log(f"{stage}/mIoU", miou, prog_bar=True, sync_dist=True)
        return loss

    def training_step(self, batch, _):
        return self._step(batch, "train")

    def validation_step(self, batch, _):
        self._step(batch, "val")

    def configure_optimizers(self):
        opt = torch.optim.Adam(
            self.parameters(),
            lr=self.cfg.train.lr,
            weight_decay=self.cfg.train.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.cfg.train.epochs
        )
        return [opt], [scheduler]
