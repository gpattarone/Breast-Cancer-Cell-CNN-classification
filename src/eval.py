# -*- coding: utf-8 -*-
"""
eval.py - Evaluation script for Breast Cancer Cell Classifier
-------------------------------------------------------------
Evaluates a trained ResNet18 model on the test dataset.
Generates metrics and saves plots.
"""

import argparse
import yaml
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import seaborn as sns

from src.models.cnn import build_resnet18
from src.data import get_dataloaders


# ---------------------------------------------------
# Evaluation
# ---------------------------------------------------

def evaluate(model, dataloader, device, class_names, out_dir="assets"):
    model.eval()
    y_true, y_pred, y_prob = [], [], []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            _, preds = torch.max(outputs, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_prob.extend(probs.cpu().numpy())

    y_true, y_pred, y_prob = np.array(y_true), np.array(y_pred), np.array(y_prob)

    # Metrics
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    plt.savefig(Path(out_dir) / "confusion_matrix.png", dpi=300)
    plt.close()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.savefig(Path(out_dir) / "roc_curve.png", dpi=300)
    plt.close()

    print(f"ROC AUC: {roc_auc:.3f}")
    print(f"Plots saved to {out_dir}/")


# ---------------------------------------------------
# Main
# ---------------------------------------------------

def main(config_path="configs/default.yaml", checkpoint="runs/best_model.pt"):
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

    # Model
    model = build_resnet18(
        num_classes=len(class_names),
        pretrained=False,
        dropout=cfg["model"]["dropout"]
    )
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model = model.to(device)

    # Eval
    evaluate(model, dataloaders["test"], device, class_names, out_dir="assets")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, default="runs/best_model.pt")
    args = parser.parse_args()
    main(args.config, args.checkpoint)
