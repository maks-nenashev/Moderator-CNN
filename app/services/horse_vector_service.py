import torch
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
from PIL import Image


class HorseVectorService:
    """
    Первая стадия каскада: быстрая генерация глобальных векторных дескрипторов 
    на базе самообученной модели DINOv2 (ViT-S/14, 384-dim).
    """

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        from app.utils.model_loader import load_local_dinov2

        self.model = load_local_dinov2("dinov2_vits14", device=self.device)

        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def extract_embedding(self, crop_image: Image.Image) -> np.ndarray:
        """
        Генерирует L2-нормированный вектор длинной 384.
        """
        tensor = self.transform(crop_image.convert("RGB")).unsqueeze(0).to(self.device)

        # Автоматическая проверка кратности 14 на случай изменения параметров Resize
        _, _, h, w = tensor.shape
        if h % 14 != 0 or w % 14 != 0:
            target_h = ((h + 13) // 14) * 14
            target_w = ((w + 13) // 14) * 14
            tensor = F.interpolate(tensor, size=(target_h, target_w), mode="bilinear", align_corners=False)

        with torch.no_grad():
            embedding = self.model(tensor)
            embedding = F.normalize(embedding, dim=1)

        return embedding.cpu().numpy()[0]