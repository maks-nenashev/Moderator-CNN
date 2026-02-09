import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from pathlib import Path

# Конфигурация
DATA_DIR = Path("data/train")
WEIGHTS_DIR = Path("weights")
BATCH_SIZE = 8
EPOCHS = 5
LEARNING_RATE = 0.001

def train():
    WEIGHTS_DIR.mkdir(exist_ok=True)
    
    # 1. Подготовка данных
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    dataset = datasets.ImageFolder(str(DATA_DIR), transform=transform)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    num_classes = len(dataset.classes)
    
    print(f"✅ Данные загружены. Классы: {dataset.classes}")
    
    # 2. Инициализация модели (EfficientNet-B0)
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    
    # Заменяем классификатор под наше количество классов
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    
    device = torch.device("cpu") # Обучаем на CPU
    model.to(device)
    
    # 3. Функция потерь и оптимизатор
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 4. Цикл обучения
    print("🚀 Старт обучения...")
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
            if (i + 1) % 5 == 0:
                print(f"Эпоха [{epoch+1}/{EPOCHS}], Батч [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
        
        print(f"📊 Итог эпохи {epoch+1}: Средний Loss: {running_loss/len(train_loader):.4f}")

    # 5. Сохранение весов
    torch.save(model.state_dict(), WEIGHTS_DIR / "moderator_v1.pth")
    print(f"💾 Модель сохранена в {WEIGHTS_DIR}/moderator_v1.pth")

if __name__ == "__main__":
    train()
