import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import sys
from pathlib import Path

# Порядок классов должен СТРОГО совпадать с папками обучения
CLASSES = ['explicit', 'safe', 'violence']
MODEL_PATH = "weights/moderator_v1.pth"

def predict(img_path):
    # 1. Сборка архитектуры
    model = models.efficientnet_b0()
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, len(CLASSES))
    
    # 2. Загрузка весов (на CPU для Latitude)
    if not Path(MODEL_PATH).exists():
        print(f"❌ Файл весов {MODEL_PATH} не найден!")
        return

    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()
    
    # 3. Трансформация (как при обучении)
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    try:
        img = Image.open(img_path).convert('RGB')
        input_tensor = preprocess(img).unsqueeze(0)
        
        # 4. Вычисление
        with torch.no_grad():
            output = model(input_tensor)
            prob = torch.nn.functional.softmax(output[0], dim=0)
            conf, idx = torch.max(prob, 0)
        
        print(f"🔍 ВЕРДИКТ: {CLASSES[idx].upper()} (Уверенность: {conf.item()*100:.2f}%)")
    except Exception as e:
        print(f"❌ Ошибка при чтении файла: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("💡 Использование: python3 scripts/predict.py путь/к/картинке.jpg")
    else:
        predict(sys.argv[1])
