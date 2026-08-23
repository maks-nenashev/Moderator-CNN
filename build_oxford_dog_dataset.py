import os
import tarfile
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
import shutil
import random

BASE_DIR = Path('/home/maks/Moderator-CNN/data/dog_features')
TMP_DIR = Path('/tmp/oxford_pets')

# 1. Очистка старых данных
if BASE_DIR.exists():
    shutil.rmtree(BASE_DIR)
if TMP_DIR.exists():
    shutil.rmtree(TMP_DIR)

TMP_DIR.mkdir(parents=True, exist_ok=True)

# 2. Скачивание с серверов Oxford University
IMAGES_URL = 'https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz'
ANNOTATIONS_URL = 'https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz'

images_tar = TMP_DIR / 'images.tar.gz'
annots_tar = TMP_DIR / 'annotations.tar.gz'

print("📥 Скачивание изображений с серверов Oxford...")
urllib.request.urlretrieve(IMAGES_URL, images_tar)

print("📥 Скачивание аннотаций морд/голов...")
urllib.request.urlretrieve(ANNOTATIONS_URL, annots_tar)

print("📦 Распаковка архивов...")
with tarfile.open(images_tar, 'r:gz') as tar:
    tar.extractall(TMP_DIR)

with tarfile.open(annots_tar, 'r:gz') as tar:
    tar.extractall(TMP_DIR)

# 3. Фильтрация строго собак (Species_ID == 2 в list.txt)
dog_names = set()
list_file = TMP_DIR / 'annotations' / 'list.txt'

with open(list_file, 'r') as f:
    for line in f:
        if line.startswith('#'):
            continue
        parts = line.strip().split()
        if len(parts) >= 3:
            img_name, class_id, species_id = parts[0], parts[1], parts[2]
            if species_id == '2':  # 2 = Dog (1 = Cat)
                dog_names.add(img_name)

print(f"🐶 Отфильтровано собак: {len(dog_names)}")

# 4. Формирование структуры YOLOv8
for split in ['train', 'valid']:
    (BASE_DIR / split / 'images').mkdir(parents=True, exist_ok=True)
    (BASE_DIR / split / 'labels').mkdir(parents=True, exist_ok=True)

xml_dir = TMP_DIR / 'annotations' / 'xmls'
images_dir = TMP_DIR / 'images'

items = []
for img_name in dog_names:
    xml_path = xml_dir / f"{img_name}.xml"
    img_path = images_dir / f"{img_name}.jpg"
    if xml_path.exists() and img_path.exists():
        items.append((img_name, xml_path, img_path))

random.seed(42)
random.shuffle(items)

split_idx = int(len(items) * 0.8)
train_items = items[:split_idx]
valid_items = items[split_idx:]

def convert_to_yolo(item_list, split_name):
    count = 0
    for img_name, xml_path, img_path in item_list:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        size = root.find('size')
        if size is None:
            continue
        w = float(size.find('width').text)
        h = float(size.find('height').text)
        if w == 0 or h == 0:
            continue
            
        yolo_labels = []
        for obj in root.findall('object'):
            bndbox = obj.find('bndbox')
            if bndbox is not None:
                xmin = float(bndbox.find('xmin').text)
                ymin = float(bndbox.find('ymin').text)
                xmax = float(bndbox.find('xmax').text)
                ymax = float(bndbox.find('ymax').text)
                
                # Конвертация в YOLO format (cx, cy, bw, bh normalized)
                bw = (xmax - xmin) / w
                bh = (ymax - ymin) / h
                cx = (xmin + (xmax - xmin) / 2.0) / w
                cy = (ymin + (ymax - ymin) / 2.0) / h
                
                # Ограничение значений в диапазоне [0, 1]
                cx, cy = max(0.0, min(1.0, cx)), max(0.0, min(1.0, cy))
                bw, bh = max(0.0, min(1.0, bw)), max(0.0, min(1.0, bh))
                
                yolo_labels.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        
        if yolo_labels:
            shutil.copy(img_path, BASE_DIR / split_name / 'images' / f"{img_name}.jpg")
            with open(BASE_DIR / split_name / 'labels' / f"{img_name}.txt", 'w') as lf:
                lf.write('\n'.join(yolo_labels))
            count += 1
    return count

n_train = convert_to_yolo(train_items, 'train')
n_val = convert_to_yolo(valid_items, 'valid')

# 5. Генерация манифеста data.yaml
data_yaml_content = f"""path: {BASE_DIR}
train: train/images
val: valid/images

nc: 1
names: ['dog_head']
"""

with open(BASE_DIR / 'data.yaml', 'w') as f:
    f.write(data_yaml_content)

print(f"✅ Готово. Сформирован датасет: {n_train} train, {n_val} valid кадров.")
print(f"📄 Манифест записан в {BASE_DIR / 'data.yaml'}")

# Очистка временных файлов
shutil.rmtree(TMP_DIR)
