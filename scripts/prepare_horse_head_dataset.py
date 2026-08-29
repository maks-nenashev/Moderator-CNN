import os
import re
import shutil
import random
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "datasets" / "raw_horse_heads"
OUT_DIR = BASE_DIR / "datasets" / "horse_head_local"

VAL_RATIO = 0.2
SEED = 42

def normalize_name(stem: str) -> str:
    """Приводит имена файлов к единому стандарту, убирая хэши Label Studio и нормализуя дубликаты."""
    # 1. Удаление хэша Label Studio
    if "-" in stem:
        parts = stem.split("-", 1)
        if len(parts[0]) in (8, 36):
            stem = parts[1]
    # 2. Приведение ' (1)' к '_1' и замена пробелов
    stem = re.sub(r'\s*\(\s*(\d+)\s*\)', r'_\1', stem)
    stem = stem.replace(" ", "_")
    return stem

def prepare_dataset():
    if not RAW_DIR.exists():
        print(f"🛑 Ошибка: Папка {RAW_DIR} не найдена.")
        return

    valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
    
    # Индексация картинок с нормализацией ключей
    raw_imgs = {}
    for p in RAW_DIR.glob("*"):
        if p.suffix.lower() in valid_exts:
            key = normalize_name(p.stem)
            raw_imgs[key] = p
    
    # Индексация меток с нормализацией ключей
    raw_lbls = {}
    for p in RAW_DIR.glob("*.txt"):
        if p.name == "classes.txt":
            continue
        key = normalize_name(p.stem)
        raw_lbls[key] = p

    # Поиск пересечений по нормализованным ключам
    common_keys = sorted(list(set(raw_imgs.keys()) & set(raw_lbls.keys())))

    if not common_keys:
        print("🛑 Ошибка: Не найдено пар картинка+разметка!")
        return

    print(f"📦 Успешно сопоставлено пар: {len(common_keys)} из {max(len(raw_imgs), len(raw_lbls))}")

    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)

    for split in ["train", "val"]:
        (OUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    random.seed(SEED)
    random.shuffle(common_keys)

    val_count = int(len(common_keys) * VAL_RATIO)
    val_keys = set(common_keys[:val_count])

    counts = {"train": 0, "val": 0}

    for key in common_keys:
        split = "val" if key in val_keys else "train"
        img_src = raw_imgs[key]
        lbl_src = raw_lbls[key]

        # Сохранение с унифицированным именем (без скобок и хэшей)
        clean_filename = f"{key}{img_src.suffix.lower()}"
        clean_labelname = f"{key}.txt"

        shutil.copy2(img_src, OUT_DIR / "images" / split / clean_filename)
        shutil.copy2(lbl_src, OUT_DIR / "labels" / split / clean_labelname)
        
        counts[split] += 1

    yaml_content = f"""path: {OUT_DIR.absolute()}
train: images/train
val: images/val

names:
  0: horse_head
"""
    with open(OUT_DIR / "data.yaml", "w") as f:
        f.write(yaml_content)

    print(f"✅ Датасет готов в {OUT_DIR}")
    print(f"📊 Train: {counts['train']} пар | Val: {counts['val']} пар")

if __name__ == "__main__":
    prepare_dataset()
