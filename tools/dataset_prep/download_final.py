import os
import zipfile
import requests
from pathlib import Path

def download_file(url, save_path):
    print(f"📥 Загрузка: {url}")
    r = requests.get(url, stream=True)
    with open(save_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)

def main():
    # Создаем структуру
    base_path = Path("data/train")
    base_path.mkdir(parents=True, exist_ok=True)

    # Прямая ссылка на проверенный мини-сет (NSFW + Safe)
    # Этот архив обычно доступен в репозиториях по фильтрации
    url = "https://github.com/alexkimxyz/nsfw_data_scrapper/raw/master/data.zip"
    zip_path = "nsfw_data.zip"

    try:
        download_file(url, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall("data/tmp_data")
        
        # Переносим (пути внутри архива)
        os.system("mv data/tmp_data/data/train/porn/* data/train/explicit/ 2>/dev/null")
        os.system("mv data/tmp_data/data/train/neutral/* data/train/safe/ 2>/dev/null")
        
        print("✅ Explicit и Safe заполнены.")
    except Exception as e:
        print(f"❌ Ошибка: {e}")
    finally:
        if os.path.exists(zip_path): os.remove(zip_path)

if __name__ == "__main__":
    main()
