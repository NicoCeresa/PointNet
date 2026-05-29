import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from pointnet import ModelNet40, PointNetCls

DATA_ROOT = "../datasets/ModelNet40"
CHECKPOINT = "checkpoints/pointnet_cls.pt"
NUM_POINTS = 1024
BATCH_SIZE = 32
EPOCHS = 200
LR = 1e-3


def accuracy(logits, labels):
    return (logits.argmax(1) == labels).float().mean().item()


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for pts, labels in loader:
            pts, labels = pts.to(device), labels.to(device)
            correct += (model(pts).argmax(1) == labels).sum().item()
            total += len(labels)
    return correct / total


def train(data_root: str, checkpoint: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    train_ds = ModelNet40(data_root, split="train", num_points=NUM_POINTS, augment=True)
    test_ds  = ModelNet40(data_root, split="test",  num_points=NUM_POINTS, augment=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model     = PointNetCls(num_classes=len(train_ds.classes)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    os.makedirs(os.path.dirname(checkpoint), exist_ok=True)
    best_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for pts, labels in train_loader:
            pts, labels = pts.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(pts), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)
        scheduler.step()

        test_acc = evaluate(model, test_loader, device)
        print(f"epoch {epoch:3d}/{EPOCHS}  loss {total_loss/len(train_ds):.4f}  test {test_acc:.3f}")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({"state_dict": model.state_dict(), "classes": train_ds.classes}, checkpoint)
            print(f"  → checkpoint saved  (best: {best_acc:.3f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",       default=DATA_ROOT,   help="path to ModelNet40 root")
    parser.add_argument("--checkpoint", default=CHECKPOINT,  help="where to save the best model")
    args = parser.parse_args()
    train(args.data, args.checkpoint)
