import io
import logging
import os
import numpy as np
from PIL import Image, ImageOps
import torch
from torchvision import transforms
from ultralytics import YOLO

from app.models.dog_biometrics import DogBiometricNet

logger = logging.getLogger(__name__)

# Рабочий порог L2 для ArcFace (Match <= 0.38)
MATCH_THRESHOLD = 0.65


def pad_to_square(image: Image.Image) -> Image.Image:
    """Сохранение пропорций кадра за счет добавления симметричных полей (letterboxing)."""
    w, h = image.size
    max_side = max(w, h)
    hp = (max_side - w) // 2
    vp = (max_side - h) // 2
    padding = (hp, vp, max_side - (w + hp), max_side - (h + vp))
    return ImageOps.expand(image, padding, fill=(0, 0, 0))


INFERENCE_TRANSFORMS = transforms.Compose([
    transforms.Lambda(pad_to_square),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    ),
])


class DogBiometricsService:

    def __init__(
        self,
        embedder_weights_path: str,
        yolo_weights_path: str,
        device: str = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # 1. Загрузка детектора объектов YOLO
        if not os.path.exists(yolo_weights_path):
            raise FileNotFoundError(f"YOLO weights missing: {yolo_weights_path}")

        self.yolo_model = YOLO(yolo_weights_path)

        # 2. Инициализация сети ArcFace ResNet34
        self.embedder = DogBiometricNet(
            backbone_name="resnet34",
            embedding_size=512,
            pretrained=False,
        )

        # 3. Каноническая нормализация и сопоставление весов
        if os.path.exists(embedder_weights_path):
            try:
                checkpoint = torch.load(
                    embedder_weights_path, map_location=self.device
                )

                # Рекурсивный поиск словаря с PyTorch-тензорами
                state_dict = None
                if isinstance(checkpoint, dict):
                    if any(isinstance(v, torch.Tensor) for v in checkpoint.values()):
                        state_dict = checkpoint
                    else:
                        for _, v in checkpoint.items():
                            if isinstance(v, dict) and any(
                                isinstance(tv, torch.Tensor) for tv in v.values()
                            ):
                                state_dict = v
                                break

                if state_dict is None:
                    state_dict = checkpoint if isinstance(checkpoint, dict) else {}

                # Функция очистки служебных префиксов
                def normalize_key(key: str) -> str:
                    for prefix in ("module.", "_orig_mod.", "backbone."):
                        if key.startswith(prefix):
                            key = key[len(prefix):]
                    return key

                model_state = self.embedder.state_dict()
                # Индекс оригинальных ключей модели по их нормализованной форме
                model_norm_map = {
                    normalize_key(k): k for k in model_state.keys()
                }

                adapted_state = {}
                for file_k, tensor in state_dict.items():
                    if not isinstance(tensor, torch.Tensor):
                        continue
                    norm_k = normalize_key(file_k)
                    if norm_k in model_norm_map:
                        target_k = model_norm_map[norm_k]
                        if model_state[target_k].shape == tensor.shape:
                            adapted_state[target_k] = tensor

                missing_keys, unexpected_keys = self.embedder.load_state_dict(
                    adapted_state, strict=False
                )

                mapped_count = len(adapted_state)
                total_count = len(model_state)

                if mapped_count < total_count:
                    logger.warning(
                        f"⚠️ Частичная загрузка весов [{embedder_weights_path}]: "
                        f"{mapped_count}/{total_count} слоев сопоставлено. "
                        f"Пропущено: {len(missing_keys)}"
                    )
                else:
                    logger.info(
                        f"✅ ArcFace Embedder (ResNet34) успешно загружен из [{embedder_weights_path}]: "
                        f"{mapped_count}/{total_count} слоев."
                    )

            except Exception as e:
                logger.error(f"❌ Критическая ошибка загрузки весов: {e}")
                raise e
        else:
            logger.warning(f"⚠️ Файл весов не найден: {embedder_weights_path}")

        self.embedder.to(self.device)
        self.embedder.eval()

    def process_image(
        self, image_bytes: bytes, conf_threshold: float = 0.35
    ) -> dict:
        """Основной метод инференса: детекция -> EXIF авто-разворот -> квадратная центровка BBox -> экспорт L2-нормализованного вектора."""
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            image = ImageOps.exif_transpose(image)
        except Exception as e:
            raise ValueError(f"Invalid image format: {str(e)}")

        img_w, img_h = image.size

        # Детекция головы через YOLO
        results = self.yolo_model(image, conf=conf_threshold, verbose=False)
        boxes = results[0].boxes

        if len(boxes) == 0:
            return {
                "status": "no_dog_detected",
                "embedding": None,
                "bbox": None,
                "confidence": 0.0,
            }

        best_box = max(boxes, key=lambda b: float(b.conf[0]))
        conf = float(best_box.conf[0])
        x1, y1, x2, y2 = map(int, best_box.xyxy[0].tolist())

        # Квадратная центровка BBox с 15% запасом
        w, h = x2 - x1, y2 - y1
        cx, cy = x1 + w // 2, y1 + h // 2
        side = int(max(w, h) * 1.15)

        x1_s = max(0, cx - side // 2)
        y1_s = max(0, cy - side // 2)
        x2_s = min(img_w, cx + side // 2)
        y2_s = min(img_h, cy + side // 2)

        crop_w = x2_s - x1_s
        crop_h = y2_s - y1_s

        # Отбраковка полнокадровых кропов (>90%)
        if (crop_w * crop_h) >= (img_w * img_h * 0.90):
            return {
                "status": "full_frame_rejected",
                "embedding": None,
                "bbox": None,
                "confidence": round(conf, 4),
            }

        cropped = image.crop((x1_s, y1_s, x2_s, y2_s))
        tensor = INFERENCE_TRANSFORMS(cropped).unsqueeze(0).to(self.device)

        with torch.no_grad():
            embedding_tensor = self.embedder.extract_features(tensor)

        emb_array = embedding_tensor.squeeze(0).cpu().numpy()

        return {
            "status": "success",
            "embedding": emb_array.tolist(),
            "bbox": {"x": x1_s, "y": y1_s, "w": crop_w, "h": crop_h},
            "confidence": round(conf, 4),
        }

    def predict_embedding(
        self, image_bytes: bytes, bbox: list[int] = None
    ) -> dict:
        return self.process_image(image_bytes)