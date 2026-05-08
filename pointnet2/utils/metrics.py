import torch


def mean_iou(preds: torch.Tensor, targets: torch.Tensor, num_classes: int) -> torch.Tensor:
    """preds/targets: (B, N) — returns scalar mean IoU"""
    raise NotImplementedError


def part_iou(preds: torch.Tensor, targets: torch.Tensor, num_classes: int) -> torch.Tensor:
    """ShapeNet eval protocol: mean IoU ignoring absent classes per shape."""
    raise NotImplementedError


def overall_accuracy(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError


def per_class_accuracy(preds: torch.Tensor, targets: torch.Tensor, num_classes: int) -> torch.Tensor:
    """returns: (num_classes,) per-class accuracy"""
    raise NotImplementedError
