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
        from app.utils.model_loader import load_local_dinov2

        self.backbone = load_local_dinov2("dinov2_vits14", device="cpu")
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Автоматическое приведение H и W к ближайшему кратному 14 (для patch_size=14)
        _, _, h, w = x.shape
        if h % 14 != 0 or w % 14 != 0:
            target_h = ((h + 13) // 14) * 14
            target_w = ((w + 13) // 14) * 14
            x = F.interpolate(x, size=(target_h, target_w), mode="bilinear", align_corners=False)

        features = self.backbone(x)
        return F.normalize(features, p=2, dim=1)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)


DogBiometricNetArcFace = DogBiometricNet