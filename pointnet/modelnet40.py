import os

import numpy as np
import torch
from torch.utils.data import Dataset


def _read_off(path: str):
    """Parse an OFF mesh file → (vertices, faces) as float32/int64 arrays."""
    with open(path) as f:
        line = f.readline().strip()
        if not line.startswith("OFF"):
            raise ValueError(f"Not an OFF file: {path}")
        counts = line[3:].strip()
        if counts:
            n_verts, n_faces, _ = map(int, counts.split())
        else:
            n_verts, n_faces, _ = map(int, f.readline().split())
        verts = np.array([f.readline().split()[:3] for _ in range(n_verts)], dtype=np.float32)
        faces = np.array([f.readline().split()[1:4] for _ in range(n_faces)], dtype=np.int64)
    return verts, faces


def _sample_surface(verts: np.ndarray, faces: np.ndarray, n: int) -> np.ndarray:
    """Sample n points uniformly from a triangle mesh surface."""
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    probs = areas / areas.sum()
    idx = np.random.choice(len(faces), size=n, p=probs)
    r1 = np.random.rand(n, 1).astype(np.float32)
    r2 = np.random.rand(n, 1).astype(np.float32)
    fold = (r1 + r2) > 1
    r1[fold], r2[fold] = 1 - r1[fold], 1 - r2[fold]
    return (1 - r1 - r2) * v0[idx] + r1 * v1[idx] + r2 * v2[idx]


def _augment(pts: np.ndarray) -> np.ndarray:
    theta = np.random.uniform(0, 2 * np.pi)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
    pts = pts @ R.T
    pts += np.clip(np.random.normal(0, 0.02, pts.shape).astype(np.float32), -0.05, 0.05)
    return pts


class ModelNet40(Dataset):
    """ModelNet40 point cloud dataset sampled from mesh surfaces."""

    def __init__(self, root: str, split: str = "train", num_points: int = 1024, augment: bool = False):
        assert split in ("train", "test"), f"split must be 'train' or 'test', got '{split}'"
        self.num_points = num_points
        self.augment = augment
        self.classes = sorted(
            d for d in os.listdir(root)
            if os.path.isdir(os.path.join(root, d, "train"))
        )
        self._class_to_idx = {c: i for i, c in enumerate(self.classes)}

        self._samples = []  # (path, label)
        for cls in self.classes:
            folder = os.path.join(root, cls, split)
            if not os.path.isdir(folder):
                continue
            label = self._class_to_idx[cls]
            for fname in sorted(os.listdir(folder)):
                if fname.endswith(".off"):
                    self._samples.append((os.path.join(folder, fname), label))

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int):
        path, label = self._samples[idx]
        verts, faces = _read_off(path)
        pts = _sample_surface(verts, faces, self.num_points)  # (N, 3)

        pts -= pts.mean(axis=0)
        pts /= np.max(np.linalg.norm(pts, axis=1))

        if self.augment:
            pts = _augment(pts)

        return torch.from_numpy(pts), label
