import io
import logging
import os
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

from app.models.dog_biometrics import DogBiometricNet

logger = logging.getLogger(__name__)

INFERENCE_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    ),
])


class DogBiometricsService:

  def __init__(self, weights_path: str, device: str = None):
    self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    custom_weights_exist = os.path.exists(weights_path)

    try:
      self.model = DogBiometricNet(
          backbone_name="efficientnet_b0",
          embedding_size=512,
          pretrained=not custom_weights_exist,
      )
    except Exception as e:
      logger.error(f"Failed to load backbone: {e}")
      self.model = DogBiometricNet(
          backbone_name="efficientnet_b0", embedding_size=512, pretrained=False
      )

    if custom_weights_exist:
      try:
        checkpoint = torch.load(weights_path, map_location=self.device)
        state_dict = checkpoint.get("state_dict", checkpoint)
        self.model.load_state_dict(state_dict)
      except Exception as e:
        logger.error(f"Failed to load weights: {e}")

    self.model.to(self.device)
    self.model.eval()

  def predict_embedding(
      self, image_bytes: bytes, bbox: list[int] = None
  ) -> dict:
    try:
      image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
      raise ValueError(f"Invalid image format: {str(e)}")

    # 1. Cropping (Если передан bbox [x, y, w, h] от YOLO)
    if bbox and len(bbox) == 4 and sum(bbox) > 0:
      x, y, w, h = bbox
      # Обрезка по рамке детекции
      image = image.crop((x, y, x + w, y + h))
    else:
      # Если bbox не передан, рамкой считается всё изображение
      bbox = [0, 0, image.width, image.height]

    # 2. Трансформация закропнутого фрагмента
    tensor = INFERENCE_TRANSFORMS(image).unsqueeze(0).to(self.device)

    # 3. Извлечение признаков
    with torch.no_grad():
      embedding_tensor = self.model.extract_features(tensor)

    emb_array = embedding_tensor.squeeze(0).cpu().numpy()

    # 4. Явная L2-нормализация (Гарантирует distance range [0, 2] в pgvector)
    norm = np.linalg.norm(emb_array)
    if norm > 0:
      emb_array = emb_array / norm

    return {"embedding": emb_array.tolist(), "bbox": bbox}