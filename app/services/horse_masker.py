import torch
import numpy as np
import cv2
from PIL import Image
from mobile_sam import sam_model_registry, SamPredictor


class HorseMasker:
    def __init__(self, checkpoint_path: str = "/home/maks/Moderator-CNN/models/horse/mobile_sam.pt"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        sam = sam_model_registry["vit_t"](checkpoint=checkpoint_path)
        sam.to(device=self.device)
        sam.eval()
        self.predictor = SamPredictor(sam)

    @torch.inference_mode()
    def mask_background(self, crop_image: Image.Image) -> Image.Image:
        img_np = np.array(crop_image.convert("RGB"))
        h, w, _ = img_np.shape

        # Ускорение инференса ViT за счет FP16/autocast
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