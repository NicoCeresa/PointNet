import torch


def mean_iou(preds: torch.Tensor, targets: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    preds:   (B, N) predicted class indices
    targets: (B, N) ground-truth class indices
    returns: scalar mean IoU over classes present in targets
    """
    ious = []
    for cls in range(num_classes):
        pred_mask = preds == cls
        tgt_mask = targets == cls
        intersection = (pred_mask & tgt_mask).sum().float()
        union = (pred_mask | tgt_mask).sum().float()
        if union == 0:
            continue
        ious.append(intersection / union)
    if not ious:
        return torch.tensor(0.0, device=preds.device)
    return torch.stack(ious).mean()


def part_iou(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """
    Computes mean IoU for part segmentation.
    Ignores classes not present in a given shape (as in the ShapeNet eval protocol).

    preds:   (B, N)
    targets: (B, N)
    returns: scalar mean IoU
    """
    B = preds.shape[0]
    batch_iou = []
    for b in range(B):
        shape_ious = []
        present = targets[b].unique()
        for cls in present:
            pred_mask = preds[b] == cls
            tgt_mask = targets[b] == cls
            intersection = (pred_mask & tgt_mask).sum().float()
            union = (pred_mask | tgt_mask).sum().float()
            shape_ious.append(intersection / union.clamp(min=1e-6))
        batch_iou.append(torch.stack(shape_ious).mean())
    return torch.stack(batch_iou).mean()


def overall_accuracy(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return (preds == targets).float().mean()


def per_class_accuracy(
    preds: torch.Tensor, targets: torch.Tensor, num_classes: int
) -> torch.Tensor:
    """returns: (num_classes,) per-class accuracy"""
    accs = []
    for cls in range(num_classes):
        mask = targets == cls
        if mask.sum() == 0:
            accs.append(torch.tensor(0.0, device=preds.device))
        else:
            accs.append((preds[mask] == cls).float().mean())
    return torch.stack(accs)
