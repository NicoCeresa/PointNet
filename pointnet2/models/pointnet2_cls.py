import torch
import torch.nn as nn
import pytorch_lightning as pl
from omegaconf import DictConfig

from pointnet2.modules import (
    PointNetSetAbstraction,
    PointNetSetAbstractionGlobal,
    PointNetSetAbstractionMsg,
)


class PointNet2Cls(nn.Module):
    """
    PointNet++ classification network.
    use_msg=True reproduces the MSG variant from the paper (~92.8% on ModelNet40).
    use_msg=False is the SSG variant (~91.9%).
    """

    def __init__(self, num_classes: int = 40, use_msg: bool = False):
        super().__init__()

        if use_msg:
            self.sa1 = PointNetSetAbstractionMsg(
                npoint=512,
                radius_list=[0.1, 0.2, 0.4],
                nsample_list=[16, 32, 128],
                in_channel=0,
                mlp_list=[[32, 32, 64], [64, 64, 128], [64, 96, 128]],
            )
            self.sa2 = PointNetSetAbstractionMsg(
                npoint=128,
                radius_list=[0.2, 0.4, 0.8],
                nsample_list=[32, 64, 128],
                in_channel=64 + 128 + 128,  # concatenated MSG output from sa1
                mlp_list=[[64, 64, 128], [128, 128, 256], [128, 128, 256]],
            )
            self.sa3 = PointNetSetAbstractionGlobal(
                in_channel=128 + 256 + 256,
                mlp=[256, 512, 1024],
            )
        else:
            self.sa1 = PointNetSetAbstraction(512, 0.2, 32, in_channel=0, mlp=[64, 64, 128])
            self.sa2 = PointNetSetAbstraction(128, 0.4, 64, in_channel=128, mlp=[128, 128, 256])
            self.sa3 = PointNetSetAbstractionGlobal(in_channel=256, mlp=[256, 512, 1024])

        self.head = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """
        xyz: (B, N, 3)
        returns: (B, num_classes) logits
        """
        new_xyz, f = self.sa1(xyz, None)
        new_xyz, f = self.sa2(new_xyz, f)
        _, f = self.sa3(new_xyz, f)
        return self.head(f)


class PointNet2ClsModule(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.model = PointNet2Cls(
            num_classes=cfg.model.num_classes,
            use_msg=cfg.model.get("use_msg", False),
        )
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, xyz):
        return self.model(xyz)

    def _step(self, batch, stage: str):
        xyz, labels = batch
        logits = self(xyz)
        loss = self.criterion(logits, labels)
        acc = (logits.argmax(dim=-1) == labels).float().mean()
        self.log(f"{stage}/loss", loss, prog_bar=True, sync_dist=True)
        self.log(f"{stage}/acc", acc, prog_bar=True, sync_dist=True)
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
