import os
import shutil
from pathlib import Path
from ultralytics import YOLO

BASE_DIR = Path('/home/maks/Moderator-CNN')
DATA_YAML = BASE_DIR / 'data/cat_features/cat_data.yaml'
OUTPUT_DIR = BASE_DIR / 'models/cat'

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Инициализация весов YOLOv8s
model = YOLO('yolov8s.pt')

# 2. Дообучение на датасете голов кошек (30 эпох)
print("🚀 Запуск обучения детектора голов кошек (cat_head)...")
results = model.train(
    data=str(DATA_YAML),
    epochs=30,
    imgsz=640,
    batch=16,
    project=str(BASE_DIR / 'runs/detect'),
    name='cat_head_yolo',
    exist_ok=True
)

# 3. Изоляция целевых весов
best_weights = BASE_DIR / 'runs/detect/cat_head_yolo/weights/best.pt'
target_weights = OUTPUT_DIR / 'cat_yolo.pt'

if best_weights.exists():
    shutil.copy(best_weights, target_weights)
    print(f"✅ Обучение завершено. Модель сохранена: {target_weights}")
else:
    print("🛑 Ошибка: финальные веса best.pt не обнаружены.")


    # train_cat_detector.py
