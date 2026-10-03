import os
from pathlib import Path
from PIL import Image, ImageFile
from tqdm import tqdm

# Разрешаем загрузку срезанных/поврежденных файлов и снимаем лимит на пиксели
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

# Подъем из tools/dataset_prep/ в корень проекта (3 уровня вверх)
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "train"

TARGET_SIZE = (224, 224)
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

def process_image(file_path: Path):
    if file_path.suffix.lower() not in VALID_EXTENSIONS:
        return False

    try:
        with Image.open(file_path) as img:
            # Конвертируем все типы (RGBA, CMYK, Palette) в чистый RGB
            img = img.convert("RGB")
            # Быстрый ресайз с высококачественной фильтрацией
            img = img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
            
            # Перезаписываем файл в формате JPEG
            target_path = file_path.with_suffix(".jpg")
            img.save(target_path, "JPEG", quality=90)
            
            # Если исходный файл был .png/.webp, удаляем старый дубликат
            if target_path != file_path and file_path.exists():
                file_path.unlink()
                
        return True
    except Exception as e:
        print(f"\n⚠️ Удаление поврежденного файла {file_path.name}: {e}")
        try:
            file_path.unlink(missing_ok=True)
        except Exception:
            pass
        return False

def main():
    if not DATA_DIR.exists():
        print(f"❌ Директория не найдена: {DATA_DIR}")
        return

    all_files = [p for p in DATA_DIR.rglob("*") if p.is_file()]
    print(f"🔍 Найдено файлов для проверки в {DATA_DIR}: {len(all_files)}")

    success_count = 0
    for file_path in tqdm(all_files, desc="Processing images"):
        if process_image(file_path):
            success_count += 1

    print(f"✅ Успешно обработано: {success_count}/{len(all_files)} файлов.")

if __name__ == "__main__":
    main()