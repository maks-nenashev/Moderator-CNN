# app/models/cat_biometrics.py
# frozen_string_literal: false
import torch
import torch.nn as nn
from torchvision.models import resnet34, ResNet34_Weights

class CatBiometricNet(nn.Module):
    """
    Изолированная архитектура эмбеддера кошачьей биометрии (ResNet34 + ArcFace Head).
    Полностью независима от модуля dog_biometrics.
    """
    def __init__(self, embedding_size: int = 512):
        super().__init__()
        # Backbone ResNet34 без базового FC-слоя
        self.backbone = resnet34(weights=None)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        # Выходной 512D проекционный слой
        self.embedding_head = nn.Linear(in_features, embedding_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        embeddings = self.embedding_head(features)
        return embeddings

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)