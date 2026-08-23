import os
from pathlib import Path
import sys

# 1. Регистрация корня проекта
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from app.models.dog_biometrics import ArcMarginProduct, DogBiometricNet

# --- Пути ---
DATA_DIR = BASE_DIR / "data" / "dog_features" / "train_classed"
SAVE_PATH = BASE_DIR / "models" / "dog" / "arcface_v1.pth"

# --- Гиперпараметры ---
BATCH_SIZE = 32
EPOCHS = 10  # Оптимизировано для первичной валидации на CPU
LEARNING_RATE = 1e-3
EMBEDDING_SIZE = 512
ARCFACE_S = 30.0
ARCFACE_M = 0.50


def get_data_loaders():
  train_transforms = transforms.Compose([
      transforms.Resize((224, 224)),
      transforms.RandomHorizontalFlip(p=0.5),
      transforms.RandomRotation(degrees=10),
      transforms.ColorJitter(brightness=0.2, contrast=0.2),
      transforms.ToTensor(),
      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
  ])

  if not DATA_DIR.exists():
    raise FileNotFoundError(f"❌ Dataset missing at {DATA_DIR}")

  dataset = datasets.ImageFolder(
      root=str(DATA_DIR), transform=train_transforms
  )

  is_cuda = torch.cuda.is_available()
  loader = DataLoader(
      dataset,
      batch_size=BATCH_SIZE,
      shuffle=True,
      num_workers=0,  # Исключает межпроцессные дедлоки на CPU
      pin_memory=is_cuda,
  )

  return loader, len(dataset.classes)


def train():
  # Задействовать все физические ядра CPU при отсутствии CUDA
  if not torch.cuda.is_available():
    torch.set_num_threads(os.cpu_count() or 4)

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"🚀 ArcFace Training | Device: {device} | Threads: {torch.get_num_threads()}")

  train_loader, num_classes = get_data_loaders()
  print(
      f"📊 Dataset: {len(train_loader.dataset)} images | {num_classes} classes"
  )

  model = DogBiometricNet(embedding_size=EMBEDDING_SIZE, pretrained=True).to(
      device
  )
  metric_fc = ArcMarginProduct(
      in_features=EMBEDDING_SIZE,
      out_features=num_classes,
      s=ARCFACE_S,
      m=ARCFACE_M,
  ).to(device)

  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.AdamW(
      list(model.parameters()) + list(metric_fc.parameters()),
      lr=LEARNING_RATE,
      weight_decay=1e-4,
  )
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
      optimizer, T_max=EPOCHS
  )

  best_loss = float("inf")
  os.makedirs(SAVE_PATH.parent, exist_ok=True)

  for epoch in range(1, EPOCHS + 1):
    model.train()
    metric_fc.train()

    running_loss = 0.0
    correct = 0
    total = 0

    # Прогресс-бар на каждый батч
    pbar = tqdm(
        train_loader, desc=f"Epoch [{epoch:02d}/{EPOCHS:02d}]", unit="batch"
    )

    for images, labels in pbar:
      images, labels = images.to(device), labels.to(device)

      optimizer.zero_grad()
      features = model(images)
      outputs = metric_fc(features, labels)

      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()

      running_loss += loss.item() * images.size(0)
      _, predicted = outputs.max(1)
      total += labels.size(0)
      correct += predicted.eq(labels).sum().item()

      pbar.set_postfix({"batch_loss": f"{loss.item():.4f}"})

    scheduler.step()

    epoch_loss = running_loss / total
    epoch_acc = (correct / total) * 100

    print(
        f"Summary Epoch {epoch:02d} | Loss: {epoch_loss:.4f} | Acc:"
        f" {epoch_acc:.2f}%"
    )

    if epoch_loss < best_loss:
      best_loss = epoch_loss
      torch.save(model.state_dict(), SAVE_PATH)
      print(f"  💾 Checkpoint updated: {SAVE_PATH} (Loss: {best_loss:.4f})")

  print("\n✅ Training Complete. Model saved.")


if __name__ == "__main__":
  train()