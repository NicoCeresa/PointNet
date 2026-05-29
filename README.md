---
title: PointNet
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 5.0.0
app_file: app.py
pinned: false
---

# PointNet

PyTorch implementation of [PointNet](https://arxiv.org/abs/1612.00593) (Qi et al., 2017) and [PointNet++](https://arxiv.org/abs/1706.02413) (Qi et al., 2017), trained on ModelNet40 for 3D point cloud classification.

The Gradio demo lets you pick a shape category and watch the model classify a real held-out point cloud in 3D.

## Structure

```
pointnet/
  pointnet.py       # PointNet models (TNet, backbone, Cls, PartSeg)
  pointnet_pp.py    # PointNet++ models (stubs)
  modelnet40.py     # ModelNet40 dataset loader
  metrics.py        # Evaluation metrics
app.py              # Gradio demo
train.py            # Training script
scripts/
  prepare_samples.py  # Copies test samples into assets/ for the demo
```

## Usage

```python
from pointnet import PointNetCls

model = PointNetCls(num_classes=40)
logits = model(xyz)  # xyz: (B, N, 3)
```

## Training

```bash
python train.py --data ../datasets/ModelNet40 --checkpoint checkpoints/pointnet_cls.pt
```

Then prepare the demo samples from the held-out test split:

```bash
python scripts/prepare_samples.py --data ../datasets/ModelNet40 --n 5
```

## Results

| Model     | Dataset    | Accuracy |
|-----------|------------|----------|
| PointNet  | ModelNet40 | TBD      |
| PointNet++| ModelNet40 | TBD      |
