# -*- coding: utf-8 -*-
"""
cnn.py - Model definitions for Breast Cancer Cell Classification
---------------------------------------------------------------
Provides CNN architectures for training. 
Currently supports ResNet18 as baseline.
"""

import torch.nn as nn
from torchvision import models


def build_resnet18(num_classes: int = 2, pretrained: bool = False, dropout: float = 0.2):
    """
    Build a ResNet18 model for classification.

    Args:
        num_classes (int): number of output classes.
        pretrained (bool): if True, load ImageNet pretrained weights.
        dropout (float): dropout probability before final classification layer.

    Returns:
        model (nn.Module): PyTorch ResNet18 model.
    """
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)

    in_feats = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_feats, num_classes)
    )

    return model

