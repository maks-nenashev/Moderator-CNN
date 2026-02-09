import torch
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
from models.backbone import get_model

class InferenceEngine:
    def __init__(self, model_path=None, num_classes=3):
        # 1. Инициализируем архитектуру
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = get_model(num_classes=num_classes)
        
        # 2. Загружаем веса (если они есть)
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        self.model.to(self.device)
        self.model.eval()  # Переводим в режим предсказания (отключает Dropout)

        # 3. Настраиваем пайплайн трансформации (Preprocessing)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)), # EfficientNet-B0 стандарт
            transforms.ToTensor(),
            # Нормализация ImageNet (обязательна для предобученных весов)
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image_path):
        try:
            # Открываем и конвертируем в RGB (важно для PNG с альфа-каналом)
            img = Image.open(image_path).convert('RGB')
            
            # Применяем трансформации и добавляем Batch-размерность (N, C, H, W)
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)

            with torch.no_grad(): # Отключаем градиенты для экономии памяти
                logits = self.model(img_tensor)
                # Превращаем логиты в вероятности (0-1)
                probs = F.softmax(logits, dim=1)
            
            return probs[0].cpu().numpy()
        except Exception as e:
            print(f"🚨 Ошибка инференса на файле {image_path}: {e}")
            return None

# Пример запуска (для отладки)
if __name__ == "__main__":
    engine = InferenceEngine() # Пока без весов, на базовом backbone
    res = engine.predict("data/raw/test.jpg")
    print(f"📊 Raw probabilities: {res}")