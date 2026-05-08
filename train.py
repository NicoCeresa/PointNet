"""
Usage:
    python train.py --config-name=train_cls
    python train.py --config-name=train_partseg
    python train.py --config-name=train_semseg model.use_msg=true
"""

import hydra
import pytorch_lightning as pl
import wandb
from omegaconf import DictConfig
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from pointnet2.models import PointNet2ClsModule, PointNet2PartSegModule, PointNet2SemSegModule

_MODULE_MAP = {
    "cls": PointNet2ClsModule,
    "partseg": PointNet2PartSegModule,
    "semseg": PointNet2SemSegModule,
}


def build_datamodule(cfg: DictConfig) -> pl.LightningDataModule:
    """Import and instantiate the appropriate DataModule lazily to avoid top-level import errors."""
    task = cfg.model.task
    if task == "cls":
        from data.modelnet40.datamodule import ModelNet40DataModule
        return ModelNet40DataModule(cfg.data)
    if task == "partseg":
        from data.shapenet.datamodule import ShapeNetDataModule
        return ShapeNetDataModule(cfg.data)
    if task == "semseg":
        from data.s3dis.datamodule import S3DISDataModule
        return S3DISDataModule(cfg.data)
    raise ValueError(f"Unknown task: {task}")


@hydra.main(config_path="configs", config_name="train_cls", version_base=None)
def main(cfg: DictConfig) -> None:
    pl.seed_everything(42, workers=True)

    module_cls = _MODULE_MAP[cfg.model.task]
    model = module_cls(cfg)
    datamodule = build_datamodule(cfg)

    logger = WandbLogger(project=cfg.logger.project, name=cfg.logger.name)

    callbacks = [
        ModelCheckpoint(
            monitor="val/loss",
            mode="min",
            save_top_k=3,
            filename="{epoch}-{val/loss:.4f}",
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    trainer = pl.Trainer(
        max_epochs=cfg.train.epochs,
        gradient_clip_val=cfg.train.grad_clip,
        log_every_n_steps=cfg.logger.log_every_n_steps,
        logger=logger,
        callbacks=callbacks,
        deterministic=True,
    )

    trainer.fit(model, datamodule=datamodule)
    wandb.finish()


if __name__ == "__main__":
    main()
