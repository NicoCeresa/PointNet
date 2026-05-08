"""
Usage:
    python eval.py --config-name=train_cls checkpoint=path/to/checkpoint.ckpt
"""

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig

from pointnet2.models import PointNet2ClsModule, PointNet2PartSegModule, PointNet2SemSegModule
from pointnet2.utils.metrics import mean_iou, overall_accuracy, per_class_accuracy

_MODULE_MAP = {
    "cls": PointNet2ClsModule,
    "partseg": PointNet2PartSegModule,
    "semseg": PointNet2SemSegModule,
}


@hydra.main(config_path="configs", config_name="train_cls", version_base=None)
def main(cfg: DictConfig) -> None:
    assert cfg.get("checkpoint"), "Pass checkpoint=<path> on the command line"

    module_cls = _MODULE_MAP[cfg.model.task]
    model = module_cls.load_from_checkpoint(cfg.checkpoint, cfg=cfg)
    model.eval()

    from train import build_datamodule
    datamodule = build_datamodule(cfg)
    datamodule.setup("test")

    trainer = pl.Trainer(logger=False, enable_checkpointing=False)
    results = trainer.test(model, datamodule=datamodule)
    print(results)


if __name__ == "__main__":
    main()
