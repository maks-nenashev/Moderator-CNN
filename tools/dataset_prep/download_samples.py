import os
import requests
from pathlib import Path

# Конфигурация
SAMPLES_PER_CLASS = 50
DATA_DIR = Path("data/train")

# Ресурсы (Исправленные ссылки)
SOURCES = {
    "safe": "https://picsum.photos/224/224",
    "explicit": "https://raw.githubusercontent.com/alexkimxyz/nsfw_data_scrapper/master/raw_data/urls_porn.txt"
}

def download_img(url, dest_folder, prefix):
    try:
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            fname = f"{prefix}_{os.urandom(3).hex()}.jpg"
            with open(dest_folder / fname, "wb") as f:
                f.write(r.content)
            return True
    except Exception:
        return False
    return False

def main():
    for label, source in SOURCES.items():
        folder = DATA_DIR / label
        folder.mkdir(parents=True, exist_ok=True)
        print(f"\n📥 Загрузка категории: {label}...")
        
        count = 0
        if label == "safe":
            # Тянем рандомные картинки для базы
            while count < SAMPLES_PER_CLASS:
                if download_img(source, folder, "safe"):
                    count += 1
                    print(f"\rПрогресс: {count}/{SAMPLES_PER_CLASS}", end="")
        else:
            # Читаем список ссылок для NSFW
            try:
                resp = requests.get(source)
                urls = resp.text.splitlines()
                for url in urls[:SAMPLES_PER_CLASS]:
                    if download_img(url, folder, "explicit"):
                        count += 1
                        print(f"\rПрогресс: {count}/{SAMPLES_PER_CLASS}", end="")
            except Exception as e:
                print(f"Ошибка доступа к манифесту: {e}")
                
    print(f"\n\n✅ Сбор данных завершен. Проверь папку {DATA_DIR}")

if __name__ == "__main__":
    main()
