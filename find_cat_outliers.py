import sys
from pathlib import Path
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from app.services.cat_biometrics_service import cat_biometrics_service

DATA_DIR = BASE_DIR / "data/Cat"
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

def find_outliers():
    embeddings_by_id = {}

    for folder in sorted(DATA_DIR.iterdir()):
        if not folder.is_dir():
            continue
        cat_id = folder.name
        images = [f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS]
        
        valid_embs = []
        for img_path in images:
            with open(img_path, "rb") as f:
                res = cat_biometrics_service.process_image(f.read())
                if res["status"] == "success":
                    valid_embs.append((img_path.name, np.array(res["embedding"])))
        if valid_embs:
            embeddings_by_id[cat_id] = valid_embs

    cat_ids = list(embeddings_by_id.keys())

    print("=== КРИТИЧЕСКИЕ ЛОЖНЫЕ СОВПАДЕНИЯ (Different Cats with L2 < 0.50) ===")
    for i in range(len(cat_ids)):
        for j in range(i + 1, len(cat_ids)):
            id1, id2 = cat_ids[i], cat_ids[j]
            for img1_name, emb1 in embeddings_by_id[id1]:
                for img2_name, emb2 in embeddings_by_id[id2]:
                    dist = np.linalg.norm(emb1 - emb2)
                    if dist < 0.50:
                        print(f"🚨 FAR Risk: [{id1}/{img1_name}] <-> [{id2}/{img2_name}] | L2 = {dist:.4f}")

    print("\n=== КРИТИЧЕСКИЕ ЛОЖНЫЕ ОТКАЗЫ (Same Cat with L2 > 1.10) ===")
    for cat_id, items in embeddings_by_id.items():
        if len(items) < 2:
            continue
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                dist = np.linalg.norm(items[i][1] - items[j][1])
                if dist > 1.10:
                    print(f"⚠️ FRR Risk: [{cat_id}/{items[i][0]}] <-> [{cat_id}/{items[j][0]}] | L2 = {dist:.4f}")

if __name__ == "__main__":
    find_outliers()
