import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from PIL import Image

# 1. Абсолютный расчет корня проекта (/home/maks/Moderator-CNN)
BASE_DIR = Path(__file__).resolve().parent.parent

# 2. Изолированные пути домена Content Safety
DATA_DIR = BASE_DIR / "data" / "train"
SAVE_DIR = BASE_DIR / "models" / "content_safety"
SAVE_PATH = SAVE_DIR / "moderator_v1.pth"

BATCH_SIZE = 8
EPOCHS = 5
LEARNING_RATE = 1e-4

def train():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Каталог данных не найден: {DATA_DIR}")
        
    # Валидация наличия файлов в категориях
    for class_dir in DATA_DIR.iterdir():
        if class_dir.is_dir():
            valid_files = [f for f in class_dir.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']]
            if not valid_files:
                print(f"⚠️ Класс {class_dir.name} пуст. Создание заглушки placeholder.jpg")
                Image.new('RGB', (224, 224), color='black').save(class_dir / "placeholder.jpg")

    # Предобработка
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    dataset = datasets.ImageFolder(str(DATA_DIR), transform=transform)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    num_classes = len(dataset.classes)
    
    print(f"✅ Датасет загружен. Классы ({num_classes}): {dataset.class_to_idx}")
    
    # Инициализация EfficientNet-B0
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    
    # Подгрузка предыдущего чекпоинта из правильной директории
    if SAVE_PATH.exists():
        print(f"🔄 Загрузка чекпоинта модератора: {SAVE_PATH}")
        try:
            model.load_state_dict(torch.load(SAVE_PATH, map_location="cpu"))
        except Exception as e:
            print(f"⚠️ Ошибка загрузки чекпоинта, старт с ImageNet: {e}")

    device = torch.device("cpu")
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print("🚀 Старт обучения модератора контента...")
    model.train()
    
    for epoch in range(EPOCHS):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if (i + 1) % 5 == 0 or (i + 1) == len(train_loader):
                print(f"Эпоха [{epoch+1}/{EPOCHS}], Батч [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
        
        print(f"📊 Итог эпохи {epoch+1}: Средний Loss: {running_loss/len(train_loader):.4f}")

    # Сохранение весов в models/content_safety/moderator_v1.pth
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"💾 Модель сохранена в: {SAVE_PATH}")

if __name__ == "__main__":
    train()