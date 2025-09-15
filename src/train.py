# -*- coding: utf-8 -*-
"""
train.py - Breast Cancer Cell Classifier
----------------------------------------
Training script for classifying live vs dead breast cancer cells 
using a ResNet18 CNN with PyTorch.

Usage:
    python src/train.py --config configs/default.yaml
"""

import argparse
import time
import copy
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from src.models.cnn import build_resnet18
from src.data import get_dataloaders


# ---------------------------------------------------
# Training loop
# ---------------------------------------------------

def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, device, num_epochs=25):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print("-" * 20)

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss, running_corrects = 0.0, 0

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s")
    print(f"Best val Acc: {best_acc:.4f}")

    model.load_state_dict(best_model_wts)
    return model


# ---------------------------------------------------
# Main
# ---------------------------------------------------

def main(config_path="configs/default.yaml"):
    # Load config
    cfg = yaml.safe_load(open(config_path))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data
    dataloaders, dataset_sizes, class_names = get_dataloaders(
        data_root=cfg["data"]["root"],
        img_size=cfg["data"]["img_size"],
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"]
    )
    print(f"Classes: {class_names}")

    # Model
    model = build_resnet18(
        num_classes=len(class_names),
        pretrained=cfg["model"]["pretrained"],
        dropout=cfg["model"]["dropout"]
    )
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(),
                          lr=cfg["train"]["lr"],
                          momentum=0.9,
                          weight_decay=cfg["train"]["weight_decay"])

    # Train
    model = train_model(model,
                        dataloaders,
                        dataset_sizes,
                        criterion,
                        optimizer,
                        device,
                        num_epochs=cfg["train"]["epochs"])

    # Save best model
    out_dir = Path(cfg["paths"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "best_model.pt")
    print(f"Model saved to {out_dir/'best_model.pt'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()
    main(args.config)
