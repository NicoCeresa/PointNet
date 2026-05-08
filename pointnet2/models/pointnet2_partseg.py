import torch
import torch.nn as nn
import pytorch_lightning as pl
from omegaconf import DictConfig

from pointnet2.modules import (
    PointNetFeaturePropagation,
    PointNetSetAbstraction,
    PointNetSetAbstractionGlobal,
)
from pointnet2.utils.metrics import part_iou


# ShapeNet Part: 16 object categories, 50 part labels total
NUM_CATEGORIES = 16
NUM_PART_LABELS = 50


class PointNet2PartSeg(nn.Module):
    """
    PointNet++ part segmentation network (SSG).
    Category label is one-hot encoded and injected at the global feature bottleneck.
    """

    def __init__(self, num_classes: int = NUM_PART_LABELS, num_categories: int = NUM_CATEGORIES):
        super().__init__()
        self.sa1 = PointNetSetAbstraction(512, 0.2, 32, in_channel=0, mlp=[64, 64, 128])
        self.sa2 = PointNetSetAbstraction(128, 0.4, 64, in_channel=128, mlp=[128, 128, 256])
        self.sa3 = PointNetSetAbstractionGlobal(in_channel=256, mlp=[256, 512, 1024])

        # Category one-hot (16-dim) is concatenated before FP layers
        self.fp3 = PointNetFeaturePropagation(in_channel=1024 + 16 + 256, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=256 + 128, mlp=[256, 128])
        self.fp1 = PointNetFeaturePropagation(in_channel=128 + 0, mlp=[128, 128, 128])

        self.head = nn.Sequential(
            nn.Conv1d(128, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv1d(128, num_classes, 1),
        )

    def forward(self, xyz: torch.Tensor, category: torch.Tensor) -> torch.Tensor:
        """
        xyz:      (B, N, 3)
        category: (B,) integer category indices
        returns:  (B, N, num_classes) logits
        """
        B, N, _ = xyz.shape

        # Encoder
        xyz1, f1 = self.sa1(xyz, None)
        xyz2, f2 = self.sa2(xyz1, f1)
        _, f3 = self.sa3(xyz2, f2)  # (B, 1024)

        # Inject category one-hot into global features
        cat_onehot = torch.zeros(B, NUM_CATEGORIES, device=xyz.device)
        cat_onehot.scatter_(1, category.unsqueeze(1), 1.0)
        f3 = torch.cat([f3, cat_onehot], dim=-1)  # (B, 1040)

        # Treat global feature as a single virtual point at the cloud centroid
        global_xyz = xyz2.mean(dim=1, keepdim=True)  # (B, 1, 3)
        global_feat = f3.unsqueeze(1)                # (B, 1, 1040)

        f = self.fp3(xyz2, global_xyz, f2, global_feat)
        f = self.fp2(xyz1, xyz2, f1, f)
        f = self.fp1(xyz, xyz1, None, f)

        f = f.transpose(1, 2)   # (B, 128, N)
        return self.head(f).transpose(1, 2)  # (B, N, num_classes)


class PointNet2PartSegModule(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.model = PointNet2PartSeg(
            num_classes=cfg.model.num_classes,
            num_categories=cfg.model.num_categories,
        )
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, xyz, category):
        return self.model(xyz, category)

    def _step(self, batch, stage: str):
        xyz, category, labels = batch
        logits = self(xyz, category)                         # (B, N, num_classes)
        loss = self.criterion(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
        preds = logits.argmax(dim=-1)
        miou = part_iou(preds, labels, self.cfg.model.num_classes)
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
