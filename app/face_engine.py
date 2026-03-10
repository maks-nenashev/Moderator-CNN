import cv2
import numpy as np
from pathlib import Path
from insightface.app import FaceAnalysis

# Путь к твоим весам
BASE_DIR = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = BASE_DIR / "weights"

class FaceEngine:
    def __init__(self):
        # 1. Меняем L на S (Light), так как на диске сейчас S
        # 2. Указываем root, чтобы не качало в ~/.insightface
        # 3. intra_op_num_threads=2 — Risk Control для твоего i5
        self.app = FaceAnalysis(
            name='buffalo_s', 
            root=str(WEIGHTS_DIR), 
            providers=['CPUExecutionProvider'],
            provider_options=[{'intra_op_num_threads': 2}]
        )
        self.app.prepare(ctx_id=-1, det_size=(640, 640))
        print("✅ [ENGINE] Synchronized with buffalo_s. Ready.")

    def extract_from_bytes(self, image_bytes):
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None: return []
        return self._process(img)

    def _process(self, img):
        faces = self.app.get(img)
        results = []
        for face in faces:
            arc_emb = face.embedding # Это 512-d
            
            # Твоя математика: 512 + 738 = 1250
            padding = np.zeros(738)
            combined = np.concatenate([arc_emb, padding])
            
            results.append({
                "bbox": face.bbox.astype(int).tolist(),
                "embedding": combined.tolist(), # Строго 1250-d
                "conf": float(face.det_score)
            })
        return results