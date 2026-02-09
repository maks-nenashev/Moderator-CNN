import torch
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
from models.backbone import get_model

class InferenceEngine:
    def __init__(self, model_path=None, num_classes=3):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = get_model(num_classes=num_classes)
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image_path):
        try:
            img = Image.open(image_path).convert('RGB')
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self.model(img_tensor)
                probs = F.softmax(logits, dim=1)
            return probs[0].cpu().numpy()
        except Exception as e:
            print(f"🚨 Error: {e}")
            return None
