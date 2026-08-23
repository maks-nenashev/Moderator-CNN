import os
import sys
from pathlib import Path
from PIL import Image
import torch

BASE_DIR = Path(__file__).resolve().parent

if len(sys.argv) > 1:
    WEIGHTS_PATH = Path(sys.argv[1]).resolve()
else:
    DEFAULT_PATHS = [
        BASE_DIR / "weights" / "yolo_dog.pt",
        BASE_DIR / "app" / "weights" / "yolo_dog.pt",
        BASE_DIR / "weights" / "best.pt",
        BASE_DIR / "yolov8n.pt"
    ]
    WEIGHTS_PATH = next((p for p in DEFAULT_PATHS if p.exists()), DEFAULT_PATHS[0])

if len(sys.argv) > 2:
    IMAGE_PATH = Path(sys.argv[2]).resolve()
else:
    DEFAULT_IMAGES = [
        Path("/tmp/debug_inference_input.jpg"),
        BASE_DIR / "test.jpg",
        BASE_DIR / "debug_last_crop.jpg"
    ]
    IMAGE_PATH = next((p for p in DEFAULT_IMAGES if p.exists()), DEFAULT_IMAGES[0])

print("=== 1. ДИАГНОСТИКА ОКРУЖЕНИЯ И ПУТЕЙ ===")
print(f"Project Root (BASE_DIR): {BASE_DIR}")
print(f"Python Executable:       {sys.executable}")
print(f"PyTorch Version:         {torch.__version__}")
print(f"Weights File Exist?      {WEIGHTS_PATH.exists()} --> {WEIGHTS_PATH}")
print(f"Test Image Exist?        {IMAGE_PATH.exists()} --> {IMAGE_PATH}")

if not WEIGHTS_PATH.exists():
    print("\n🛑 ОШИБКА: Файл весов не найден. Передай путь явно:")
    print(f"   python test_yolo.py /absolute/path/to/weights.pt /absolute/path/to/image.jpg")
    sys.exit(1)

if not IMAGE_PATH.exists():
    print("\n🛑 ОШИБКА: Тестовое изображение не найдено. Передай путь явно вторым аргументом.")
    sys.exit(1)

from ultralytics import YOLO

print("\n=== 2. ЗАГРУЗКА И ИНФЕРЕНС YOLO ===")
model = YOLO(str(WEIGHTS_PATH))

image = Image.open(IMAGE_PATH).convert("RGB")
print(f"Размер исходного кадра: {image.width}x{image.height} px")

results = model(image, conf=0.1)

print("\n=== 3. РЕЗУЛЬТАТЫ ДЕТЕКЦИИ ===")
boxes = results[0].boxes

if len(boxes) == 0:
    print("❌ YOLO не нашла ни одного объекта при conf=0.1")
else:
    for i, box in enumerate(boxes):
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        xyxy = [round(x, 1) for x in box.xyxy[0].tolist()]
        xywh = [round(x, 1) for x in box.xywh[0].tolist()]
        print(f"Объект #{i+1}: Class ID={cls_id} | Conf={conf:.4f} | BBox XYWH={xywh} | BBox XYXY={xyxy}")
