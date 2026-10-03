import cv2
import numpy as np
from insightface.app import FaceAnalysis

class FaceEngine:
    def __init__(self):
        # Используем Buffalo_L — тяжелая, но точная модель для CPU
        self.app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        # det_thresh=0.4 — повышаем чувствительность "зрения"
        self.app.prepare(ctx_id=-1, det_size=(640, 640))

    def extract_from_bytes(self, image_bytes):
        """Метод для работы напрямую с потоком из Rails"""
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            print("🛑 [ENGINE] Decode failed: Received empty or corrupt buffer")
            return []
            
        return self._process(img)

    def extract_anchor_vector(self, img_path):
        """Метод для тестов через локальные пути"""
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"🛑 [ENGINE] File not found: {img_path}")
            return []
        return self._process(img)

    def _process(self, img):
        """Единая логика инференса и сборки вектора"""
        # Снижаем det_thresh динамически, если нужно "дожать" результат
        faces = self.app.get(img)
        
        print(f"📸 [ENGINE] Image {img.shape[1]}x{img.shape[0]} | Detected: {len(faces)} faces")
        
        results = []
        for face in faces:
            # ArcFace выдает 512-d вектор
            arc_emb = face.embedding 
            
            # Добиваем до 1250-d (как в твоем Job), чтобы сохранить архитектурное единство
            # 1250 - 512 = 738 нулей
            padding = np.zeros(738)
            combined = np.concatenate([arc_emb, padding])
            
            results.append({
                "bbox": face.bbox.astype(int).tolist(),
                "embedding": combined.tolist(),
                "conf": float(face.det_score)
            })
        return results