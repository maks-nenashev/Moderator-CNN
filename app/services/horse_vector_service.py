import torch
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
        # Загрузка весов DINOv2 Small
        self.model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", verbose=False).to(self.device).eval()

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
        with torch.no_grad():
            embedding = self.model(tensor)
            embedding = torch.nn.functional.normalize(embedding, dim=1)
        return embedding.cpu().numpy()[0]