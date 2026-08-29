import io
import logging
from PIL import Image, ImageOps
import torch
from torchvision import transforms
from ultralytics import YOLO

from app.models.dog_biometrics import DogBiometricNet

logger = logging.getLogger(__name__)

# Рабочий порог L2 для DINOv2 (будет уточнен по результатам бенчмарка)
MATCH_THRESHOLD = 0.75


def pad_to_square(image: Image.Image) -> Image.Image:
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
        embedder_weights_path: str = None,
        yolo_weights_path: str = "/home/maks/Moderator-CNN/models/dog/dog_yolo_dual.pt",
        device: str = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.yolo_model = YOLO(yolo_weights_path)
        self.embedder = DogBiometricNet().to(self.device)
        self.embedder.eval()

    def process_image(
        self, image_bytes: bytes, conf_threshold: float = 0.35
    ) -> dict:
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            image = ImageOps.exif_transpose(image)
        except Exception as e:
            raise ValueError(f"Invalid image format: {str(e)}")

        img_w, img_h = image.size
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

        w, h = x2 - x1, y2 - y1
        cx, cy = x1 + w // 2, y1 + h // 2
        side = int(max(w, h) * 1.15)

        x1_s, y1_s = max(0, cx - side // 2), max(0, cy - side // 2)
        x2_s, y2_s = min(img_w, cx + side // 2), min(img_h, cy + side // 2)

        cropped = image.crop((x1_s, y1_s, x2_s, y2_s))
        tensor = INFERENCE_TRANSFORMS(cropped).unsqueeze(0).to(self.device)

        with torch.no_grad():
            emb_tensor = self.embedder.extract_features(tensor)

        return {
            "status": "success",
            "embedding": emb_tensor.squeeze(0).cpu().numpy().tolist(),
            "bbox": {"x": x1_s, "y": y1_s, "w": x2_s - x1_s, "h": y2_s - y1_s},
            "confidence": round(conf, 4),
        }

    def predict_embedding(
        self, image_bytes: bytes, bbox: list[int] = None
    ) -> dict:
        return self.process_image(image_bytes)


dog_service = DogBiometricsService()
