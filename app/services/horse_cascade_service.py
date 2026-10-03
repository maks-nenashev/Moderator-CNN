import time
import numpy as np
import torch
from PIL import Image
from typing import List, Dict

from app.services.horse_pipeline import HorsePipeline
from app.services.horse_vector_service import HorseVectorService


class HorseCascadeService:
    def __init__(
        self,
        detector_path: str,
        top_k: int = 3,
        match_threshold: int = 14  # Оптимизировано под Tight Crop
    ):
        self.pipeline = HorsePipeline(detector_path=detector_path, sam_checkpoint_path=None)
        self.vector_service = HorseVectorService()
        self.top_k = top_k
        self.match_threshold = match_threshold
        self.gallery_index: List[Dict] = []

    def _tight_crop(self, img: Image.Image, box, crop_margin: float = 0.15) -> Image.Image:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        w = x2 - x1
        h = y2 - y1

        cx1 = int(x1 + w * crop_margin)
        cy1 = int(y1 + h * crop_margin)
        cx2 = int(x2 - w * crop_margin)
        cy2 = int(y2 - h * crop_margin)

        return img.crop((cx1, cy1, cx2, cy2))

    def enroll_horse(self, horse_id: str, image_path: str) -> bool:
        img = Image.open(image_path).convert("RGB")
        res = self.pipeline.detector(img, conf=0.50, verbose=False)
        if not res or len(res[0].boxes) == 0:
            return False

        crop = self._tight_crop(img, res[0].boxes[0])
        embedding = self.vector_service.extract_embedding(crop)

        self.gallery_index.append({
            "horse_id": horse_id,
            "crop": crop,
            "embedding": embedding,
            "image_path": image_path
        })
        return True

    @torch.inference_mode()
    def search_1_to_n(self, query_image_path: str) -> Dict:
        if not self.gallery_index:
            return {"status": "ERROR", "message": "Галерея пуста"}

        t_start = time.time()

        # 1. YOLO Head Detection
        img = Image.open(query_image_path).convert("RGB")
        res = self.pipeline.detector(img, conf=0.50, verbose=False)
        if not res or len(res[0].boxes) == 0:
            return {"status": "ERROR", "message": "Голова не найдена"}

        query_crop = self._tight_crop(img, res[0].boxes[0])
        t_yolo = (time.time() - t_start) * 1000

        # 2. DINOv2 Feature Extraction
        t_dino_start = time.time()
        query_emb = self.vector_service.extract_embedding(query_crop)
        
        gallery_embs = np.array([item["embedding"] for item in self.gallery_index])
        cos_similarities = np.dot(gallery_embs, query_emb)
        top_k_indices = np.argsort(cos_similarities)[::-1][:min(self.top_k, len(self.gallery_index))]
        t_dino = (time.time() - t_dino_start) * 1000

        # 3. LoFTR RANSAC Verification
        t_loftr_start = time.time()
        best_horse_id = None
        max_inliers = 0

        for idx in top_k_indices:
            candidate = self.gallery_index[idx]
            cand_id = candidate["horse_id"]

            match_res = self.pipeline.biometrics_service.compare_crops(query_crop, candidate["crop"])
            inliers = match_res["inliers_count"]

            if inliers > max_inliers:
                max_inliers = inliers
                if inliers >= self.match_threshold:
                    best_horse_id = cand_id

        t_loftr = (time.time() - t_loftr_start) * 1000
        is_match = max_inliers >= self.match_threshold

        return {
            "status": "MATCH" if is_match else "NO_MATCH",
            "matched_horse_id": best_horse_id,
            "max_inliers": max_inliers,
            "threshold": self.match_threshold,
            "breakdown_ms": {
                "yolo": round(t_yolo, 1),
                "dino": round(t_dino, 1),
                "loftr": round(t_loftr, 1)
            }
        }