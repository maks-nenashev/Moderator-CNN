import os
from pathlib import Path
import torch
import numpy as np
import cv2
from PIL import Image
from mobile_sam import sam_model_registry, SamPredictor

BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_SAM_PATH = BASE_DIR / "models" / "horse" / "mobile_sam.pt"


class HorseMasker:
    def __init__(self, checkpoint_path: str | Path = None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        resolved_path = checkpoint_path or os.getenv("HORSE_SAM_PATH", DEFAULT_SAM_PATH)
        
        if not Path(resolved_path).exists():
            raise FileNotFoundError(f"MobileSAM checkpoint not found at: {resolved_path}")

        sam = sam_model_registry["vit_t"](checkpoint=str(resolved_path))
        sam.to(device=self.device)
        sam.eval()
        self.predictor = SamPredictor(sam)

    @torch.inference_mode()
    def mask_background(self, crop_image: Image.Image) -> Image.Image:
        img_np = np.array(crop_image.convert("RGB"))
        h, w, _ = img_np.shape

        with torch.cuda.amp.autocast(enabled=(self.device == "cuda")):
            self.predictor.set_image(img_np)
            input_point = np.array([[w // 2, h // 2]])
            input_label = np.array([1])

            masks, _, _ = self.predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=False,
            )

        mask = masks[0].astype(np.float32)
        soft_mask = cv2.GaussianBlur(mask, (15, 15), 0)
        soft_mask = np.expand_dims(soft_mask, axis=-1)

        masked_img = (img_np * soft_mask).astype(np.uint8)
        return Image.fromarray(masked_img)