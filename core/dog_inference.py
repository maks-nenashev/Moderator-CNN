import torch
from pathlib import Path
from PIL import Image
import torchvision.transforms as T
from models.dog_biometrics import DogBiometricNet

BASE_DIR = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = BASE_DIR / "weights"
DEFAULT_WEIGHTS_PATH = WEIGHTS_DIR / "dog_biometrics_v1.pth"


class DogBiometricEngine:
    def __init__(self, weights_path: str | Path = None, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DogBiometricNet(pretrained=False).to(self.device)
        
        target_weights = Path(weights_path) if weights_path else DEFAULT_WEIGHTS_PATH
        
        if target_weights.exists():
            state_dict = torch.load(target_weights, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print(f"✅ Dog Biometrics Engine: Loaded weights from {target_weights.name}")
        else:
            print(f"⚠️ Dog Biometrics Engine: Weights not found at {target_weights}. Running on initial weights.")
        
        self.model.eval()
        
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def get_embedding(self, image: Image.Image, bbox: list[int] | None = None) -> list[float]:
        """
        Извлекает 512D вектор из PIL Image.
        :param image: исходное PIL Изображение
        :param bbox: опциональный список координат [x1, y1, x2, y2] для кадрирования
        """
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        # Кадрирование области интереса при наличии Bounding Box
        if bbox and len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            width, height = image.size
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)
            if x2 > x1 and y2 > y1:
                image = image.crop((x1, y1, x2, y2))

        tensor = self.transform(image).unsqueeze(0).to(self.device)
        embedding = self.model(tensor)
        return embedding.squeeze(0).cpu().tolist()
