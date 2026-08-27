import sys
from pathlib import Path
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from app.services.dog_biometrics_service import dog_service

DATA_DIR = BASE_DIR / "data/Dog"
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

def evaluate():
    if not DATA_DIR.exists():
        print(f"🛑 Ошибка: Директория {DATA_DIR} не найдена.")
        sys.exit(1)

    dog_folders = sorted([d for d in DATA_DIR.iterdir() if d.is_dir()])
    
    if len(dog_folders) < 2:
        print(f"🛑 Найдено папок особей: {len(dog_folders)}.")
        sys.exit(1)

    print(f"📂 Обнаружено папок особей собак: {len(dog_folders)}\n")

    embeddings_by_id = {}
    total_images = 0

    for folder in dog_folders:
        dog_id = folder.name
        images = [
            f for f in folder.iterdir() 
            if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS
        ]
        
        valid_embs = []
        for img_path in images:
            with open(img_path, "rb") as f:
                res = dog_service.process_image(f.read())
                if res["status"] == "success":
                    valid_embs.append((img_path.name, np.array(res["embedding"])))
                    total_images += 1
                else:
                    print(f"⚠️ [{dog_id}] Пропущен кадр {img_path.name}: {res['status']}")

        if valid_embs:
            embeddings_by_id[dog_id] = valid_embs

    print(f"✅ Итого обработано снимков собак: {total_images} по {len(embeddings_by_id)} особям.\n")

    intra_distances = []
    for dog_id, items in embeddings_by_id.items():
        if len(items) < 2:
            continue
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                dist = np.linalg.norm(items[i][1] - items[j][1])
                intra_distances.append(dist)

    inter_distances = []
    dog_ids = list(embeddings_by_id.keys())
    for i in range(len(dog_ids)):
        for j in range(i + 1, len(dog_ids)):
            id1, id2 = dog_ids[i], dog_ids[j]
            for _, emb1 in embeddings_by_id[id1]:
                for _, emb2 in embeddings_by_id[id2]:
                    dist = np.linalg.norm(emb1 - emb2)
                    inter_distances.append(dist)

    print("=== ВНУТРИКЛАССОВЫЕ РАССТОЯНИЯ (L2 Same Dog) ===")
    if intra_distances:
        print(f"  • Кол-во пар: {len(intra_distances)}")
        print(f"  • Мин L2:     {np.min(intra_distances):.4f}")
        print(f"  • Среднее L2: {np.mean(intra_distances):.4f}")
        print(f"  • Макс L2:    {np.max(intra_distances):.4f}")
        print(f"  • Std Dev:    {np.std(intra_distances):.4f}")

    print("\n=== МЕЖКЛАССОВЫЕ РАССТОЯНИЯ (L2 Different Dogs / Hard Negatives) ===")
    if inter_distances:
        print(f"  • Кол-во пар: {len(inter_distances)}")
        print(f"  • Мин L2:     {np.min(inter_distances):.4f}")
        print(f"  • Среднее L2: {np.mean(inter_distances):.4f}")
        print(f"  • Макс L2:    {np.max(inter_distances):.4f}")
        print(f"  • Std Dev:    {np.std(inter_distances):.4f}")

    if intra_distances and inter_distances:
        print("\n=== ОЦЕНКА ПОРОГОВ MATCH_THRESHOLD (Safety First: FAR <= 1%) ===")
        for th in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
            far = np.mean(np.array(inter_distances) <= th) * 100
            frr = np.mean(np.array(intra_distances) > th) * 100
            print(f"  Threshold {th:.2f} | FAR (Ложное совпадение): {far:5.2f}% | FRR (Ложный отказ): {frr:5.2f}%")

if __name__ == "__main__":
    evaluate()
