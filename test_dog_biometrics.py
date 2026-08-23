import sys
from pathlib import Path
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

# Безопасный импорт конфигурации и порога
try:
    from app.core.config import DOG_EMBEDDER_WEIGHTS, DOG_YOLO_WEIGHTS
except ImportError:
    from core.config import DOG_EMBEDDER_WEIGHTS, DOG_YOLO_WEIGHTS

from app.services.dog_biometrics_service import (
    DogBiometricsService,
    MATCH_THRESHOLD,
)


def calculate_l2_distance(v1: list[float], v2: list[float]) -> float:
    a = np.array(v1, dtype=np.float32)
    b = np.array(v2, dtype=np.float32)
    return float(np.linalg.norm(a - b))


def main():
    if len(sys.argv) < 2:
        print("Использование: python3 test_dog_biometrics.py <path_img1> [path_img2]")
        sys.exit(1)

    img1_path = Path(sys.argv[1])
    img2_path = Path(sys.argv[2]) if len(sys.argv) > 2 else img1_path

    print("=== [1/3] Инициализация сервиса (YOLO + ArcFace ResNet34) ===")
    service = DogBiometricsService(
        embedder_weights_path=str(DOG_EMBEDDER_WEIGHTS),
        yolo_weights_path=str(DOG_YOLO_WEIGHTS),
    )

    with open(img1_path, "rb") as f:
        bytes1 = f.read()

    with open(img2_path, "rb") as f:
        bytes2 = f.read()

    print("\n=== [2/3] Детекция и генерация векторов ===")
    res1 = service.process_image(bytes1)
    res2 = service.process_image(bytes2)

    print(
        f"Файл 1 ({img1_path.name}): Status={res1['status']}, "
        f"BBox={res1['bbox']}, Conf={res1['confidence']}"
    )
    print(
        f"Файл 2 ({img2_path.name}): Status={res2['status']}, "
        f"BBox={res2['bbox']}, Conf={res2['confidence']}"
    )

    if res1["status"] != "success" or res2["status"] != "success":
        print("\n❌ ТЕСТ ПРЕКРАЩЕН: YOLO не смогла локализовать собаку на одном из фото.")
        sys.exit(1)

    print("\n=== [3/3] Расчет L2 Distance ===")
    dist = calculate_l2_distance(res1["embedding"], res2["embedding"])
    is_match = dist <= MATCH_THRESHOLD

    print(f"L2 Distance:     {dist:.6f}")
    print(f"Match Threshold: {MATCH_THRESHOLD}")
    print(f"Вердикт:         {'✅ MATCH (Одна собака)' if is_match else '❌ NO MATCH (Разные собаки)'}")


if __name__ == "__main__":
    main()

#    cd /home/maks/Moderator-CNN
#     ./venv/bin/python3 test_dog_biometrics.py /home/maks/Загрузки/Dogs/Rusty.jpg