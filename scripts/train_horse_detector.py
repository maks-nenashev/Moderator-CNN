import sys
import shutil
from pathlib import Path
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_YAML = BASE_DIR / "datasets" / "horse_head_local" / "data.yaml"
OUTPUT_DIR = BASE_DIR / "models" / "horse"

def train():
    if not DATA_YAML.exists():
        print(f"🛑 Файл {DATA_YAML} не найден. Убедитесь, что датасет сформирован.")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("🚀 Запуск локального обучения детектора horse_head (YOLOv8s, 40 эпох)...")

    model = YOLO("yolov8s.pt")
    model.train(
        data=str(DATA_YAML),
        epochs=40,
        imgsz=512,
        batch=8,
        project=str(OUTPUT_DIR),
        name="train_run",
        exist_ok=True,
        verbose=True
    )

    best_weights = OUTPUT_DIR / "train_run" / "weights" / "best.pt"
    target_weights = OUTPUT_DIR / "horse_yolo.pt"

    if best_weights.exists():
        shutil.copy(best_weights, target_weights)
        print(f"\n✅ Обучение завершено. Веса сохранены в: {target_weights}")
    else:
        print(f"\n⚠️ Файл best.pt не найден по пути {best_weights}")

if __name__ == "__main__":
    train()
