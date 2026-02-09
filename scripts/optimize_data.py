import os
from PIL import Image
from pathlib import Path

def optimize():
    base_dir = Path("data/train")
    target_size = (224, 224)
    
    print("🚀 Старт оптимизации датасета...")
    
    for category in os.listdir(base_dir):
        cat_path = base_dir / category
        if not cat_path.is_dir(): continue
        
        print(f"📁 Обработка {category}...")
        for img_name in os.listdir(cat_path):
            img_path = cat_path / img_name
            try:
                with Image.open(img_path) as img:
                    img = img.convert('RGB')
                    img = img.resize(target_size, Image.Resampling.LANCZOS)
                    img.save(img_path, "JPEG", quality=85, optimize=True)
            except Exception as e:
                print(f"🗑 Удаление битого файла {img_name}: {e}")
                os.remove(img_path)
                
    print("✅ Оптимизация завершена.")

if __name__ == "__main__":
    optimize()
