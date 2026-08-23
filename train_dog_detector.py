from ultralytics import YOLO

model = YOLO('yolov8s.pt')

results = model.train(
    data='/home/maks/Moderator-CNN/data/dog_features/data.yaml',
    epochs=50,
    imgsz=512,       # Снизили с 640 до 512 (нагрузка падаёт существенно)
    batch=8,         # Снизили размер пакета
    workers=2,       # Жёстко ограничили фоновые процессы загрузки данных
    save_period=1,   # СОХРАНЯТЬ ЧЕКПОИНТ КАЖДУЮ ЭПОХУ!
    name='dog_yolo_dual_model',
    project='runs/detect'
)