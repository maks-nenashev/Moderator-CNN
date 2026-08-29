import torch
import cv2
import numpy as np
import torchvision.transforms as T
from PIL import Image
from kornia.feature import LoFTR


class HorseBiometricsService:
    def __init__(self, match_threshold: int = 15):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.match_threshold = match_threshold
        
        # Загрузка outdoor-модели LoFTR
        self.matcher = LoFTR(pretrained="outdoor").to(self.device).eval()
        
        # Легкое разрешение 384x384 для скорости инференса <100ms
        self.transform = T.Compose([
            T.Resize((384, 384)),
            T.ToTensor(),
        ])

    @torch.inference_mode()
    def compare_crops(self, crop_a: Image.Image, crop_b: Image.Image) -> dict:
        """
        Геометрическое сравнение двух кропов с RANSAC-фильтрацией ложных точек.
        """
        img0 = self.transform(crop_a.convert("L")).unsqueeze(0).to(self.device)
        img1 = self.transform(crop_b.convert("L")).unsqueeze(0).to(self.device)

        input_dict = {"image0": img0, "image1": img1}

        with torch.cuda.amp.autocast(enabled=(self.device == "cuda")):
            corr = self.matcher(input_dict)

        mkpts0 = corr["keypoints0"].cpu().numpy()
        mkpts1 = corr["keypoints1"].cpu().numpy()
        confidence = corr["confidence"].cpu().numpy()

        # 1. Фильтрация низковероятных совпадений (Conf > 0.25)
        valid_mask = confidence > 0.25
        pts0 = mkpts0[valid_mask]
        pts1 = mkpts1[valid_mask]

        if len(pts0) < 4:
            return {"inliers_count": 0, "status": "NO_MATCH"}

        # 2. RANSAC-проверка геометрической согласованности
        _, inliers_mask = cv2.findHomography(pts0, pts1, cv2.RANSAC, 3.0)
        
        inliers_count = int(inliers_mask.sum()) if inliers_mask is not None else 0

        return {
            "inliers_count": inliers_count,
            "status": "MATCH" if inliers_count >= self.match_threshold else "NO_MATCH"
        }