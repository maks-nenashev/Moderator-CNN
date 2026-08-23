import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class DogBiometricNet(nn.Module):

  def __init__(
      self,
      embedding_size: int = 512,
      pretrained: bool = False,
      backbone_name: str = "resnet34",  # Фикс: поддержка ключа backbone_name
  ):
    super().__init__()
    weights = models.ResNet34_Weights.DEFAULT if pretrained else None
    self.backbone = models.resnet34(weights=weights)

    in_features = self.backbone.fc.in_features
    self.backbone.fc = nn.Sequential(
        nn.BatchNorm1d(in_features),
        nn.Dropout(p=0.2),
        nn.Linear(in_features, embedding_size, bias=False),
        nn.BatchNorm1d(embedding_size),
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    features = self.backbone(x)
    return F.normalize(features, p=2, dim=1)

  def extract_features(self, x: torch.Tensor) -> torch.Tensor:
    return self.forward(x)


# Алиас для совместимости
DogBiometricNetArcFace = DogBiometricNet