import os
import random

import numpy as np
import plotly.graph_objects as go
import gradio as gr
import torch

from pointnet import PointNetCls
from pointnet.modelnet40 import _read_off, _sample_surface

CHECKPOINT   = "checkpoints/pointnet_cls.pt"
SAMPLES_ROOT = "assets/samples"
NUM_POINTS   = 1024

# ── load model ────────────────────────────────────────────────────────────────

ckpt    = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
classes = ckpt["classes"]
model   = PointNetCls(num_classes=len(classes))
model.load_state_dict(ckpt["state_dict"])
model.eval()

# ── helpers ───────────────────────────────────────────────────────────────────

def list_samples(cls: str) -> list[str]:
    folder = os.path.join(SAMPLES_ROOT, cls)
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".off")]


def point_cloud_figure(pts: np.ndarray) -> go.Figure:
    fig = go.Figure(go.Scatter3d(
        x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
        mode="markers",
        marker=dict(size=2, color=pts[:, 2], colorscale="Viridis", opacity=0.8),
    ))
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)),
        paper_bgcolor="black",
    )
    return fig


# ── inference ─────────────────────────────────────────────────────────────────

def run(cls: str):
    path = random.choice(list_samples(cls))
    verts, faces = _read_off(path)
    pts = _sample_surface(verts, faces, NUM_POINTS)
    pts -= pts.mean(axis=0)
    pts /= np.max(np.linalg.norm(pts, axis=1))

    with torch.no_grad():
        logits = model(torch.from_numpy(pts).unsqueeze(0))   # (1, N, 3)
        probs  = torch.softmax(logits, dim=1)[0].numpy()

    top5_idx   = probs.argsort()[::-1][:5]
    top5_label = {classes[i]: float(probs[i]) for i in top5_idx}

    return point_cloud_figure(pts), top5_label


# ── interface ─────────────────────────────────────────────────────────────────

with gr.Blocks(title="PointNet — ModelNet40") as demo:
    gr.Markdown("## PointNet Classification\nPick a shape category and see the model predict it from a 3D point cloud.")

    with gr.Row():
        dropdown = gr.Dropdown(choices=classes, value=classes[0], label="Category")
        btn      = gr.Button("Run", variant="primary")

    with gr.Row():
        plot   = gr.Plot(label="Point Cloud")
        labels = gr.Label(label="Predictions", num_top_classes=5)

    btn.click(fn=run, inputs=dropdown, outputs=[plot, labels])
    dropdown.change(fn=run, inputs=dropdown, outputs=[plot, labels])

demo.launch()
