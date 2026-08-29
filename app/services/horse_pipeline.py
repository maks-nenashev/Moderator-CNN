from PIL import Image
from ultralytics import YOLO
from app.services.horse_biometrics_service import HorseBiometricsService
from app.services.horse_masker import HorseMasker


class HorsePipeline:
    def __init__(self, detector_path: str, sam_checkpoint_path: str = None):
        self.detector = YOLO(detector_path)
        self.biometrics_service = HorseBiometricsService(match_threshold=15)
        
        # Инициализация маскера, если передан путь к весам
        self.masker = HorseMasker(sam_checkpoint_path) if sam_checkpoint_path else None

    def process_verification(self, image_path_a: str, image_path_b: str) -> dict:
        img_a = Image.open(image_path_a).convert("RGB")
        img_b = Image.open(image_path_b).convert("RGB")

        # 1. Детекция голов (YOLOv8)
        res_a = self.detector(img_a, conf=0.50, verbose=False)
        res_b = self.detector(img_b, conf=0.50, verbose=False)

        if not res_a[0].boxes or not res_b[0].boxes:
            return {"error": "Голова лошади не обнаружена на одном из изображений"}

        crop_a = self._crop_head(img_a, res_a[0].boxes[0])
        crop_b = self._crop_head(img_b, res_b[0].boxes[0])

        # 2. Маскирование фона (MobileSAM)
        if self.masker:
            crop_a = self.masker.mask_background(crop_a)
            crop_b = self.masker.mask_background(crop_b)

        # 3. Спарс-матчинг (LoFTR)
        result = self.biometrics_service.compare_crops(crop_a, crop_b)
        return result

    def _crop_head(self, img: Image.Image, box) -> Image.Image:
        xyxy = box.xyxy[0].cpu().numpy().astype(int)
        x1, y1, x2, y2 = max(0, xyxy[0]), max(0, xyxy[1]), min(img.width, xyxy[2]), min(img.height, xyxy[3])
        return img.crop((x1, y1, x2, y2))