import csv
from pathlib import Path

csv_path = Path('/home/maks/Moderator-CNN/runs/detect/cat_head_yolo/results.csv')

if csv_path.exists():
    with open(csv_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        # Очистка заголовков от внутренних пробелов YOLO
        reader.fieldnames = [name.strip() for name in reader.fieldnames]
        rows = list(reader)

    if rows:
        last_epoch = rows[-1]
        map50_values = [float(r['metrics/mAP50(B)']) for r in rows if 'metrics/mAP50(B)' in r]
        best_map = max(map50_values) if map50_values else 0.0

        print("=== METRICS FOR PAPER (YOLOv8s Cat Head) ===")
        print(f"Epochs Completed: {len(rows)}")
        print(f"Precision (P):    {float(last_epoch.get('metrics/precision(B)', 0)):.4f}")
        print(f"Recall (R):       {float(last_epoch.get('metrics/recall(B)', 0)):.4f}")
        print(f"mAP@50:           {float(last_epoch.get('metrics/mAP50(B)', 0)):.4f} (Best: {best_map:.4f})")
        print(f"mAP@50-95:        {float(last_epoch.get('metrics/mAP50-95(B)', 0)):.4f}")
        print(f"Final Box Loss:   {float(last_epoch.get('val/box_loss', 0)):.4f}")
        print(f"Final Cls Loss:   {float(last_epoch.get('val/cls_loss', 0)):.4f}")
    else:
        print("Файл results.csv пуст.")
else:
    print(f"Файл {csv_path} не найден.")
