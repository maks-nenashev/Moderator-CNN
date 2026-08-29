import torch
import torch.nn as nn
import torch.nn.functional as F


class DogBiometricNet(nn.Module):
    """
    Биометрический эмбеддер собак на базе DINOv2 (ViT-S/14).
    Извлекает 384D пространственную геометрию морды без зависимости от породных меток.
    """

    def __init__(self, embedding_size: int = 384, pretrained: bool = True):
        super().__init__()
        self.backbone = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vits14"
        )
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return F.normalize(features, p=2, dim=1)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)


DogBiometricNetArcFace = DogBiometricNet
