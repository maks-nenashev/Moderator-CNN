import io
import warnings
from pathlib import Path
import cv2
from fastapi import FastAPI, File, HTTPException, UploadFile
from insightface.app import FaceAnalysis
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms

# Подавление системных предупреждений PyTorch/DINOv2/MobileSAM (xFormers/SwiGLU/Registry)
warnings.filterwarnings("ignore", category=UserWarning, module="dinov2")
warnings.filterwarnings("ignore", category=UserWarning, message=".*xFormers.*")
warnings.filterwarnings("ignore", category=UserWarning, module=".*mobile_sam.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*Overwriting.*in registry.*")

app = FastAPI(title="FindWay Master AI Engine", version="2.0.0")

# --- Настройки путей ---
BASE_DIR = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = BASE_DIR / "weights"
MODELS_DIR = BASE_DIR / "models"

MODERATOR_WEIGHTS = MODELS_DIR / "dog" / "moderator_v1.pth"
DOG_YOLO_WEIGHTS = MODELS_DIR / "dog" / "dog_yolo_dual.pt"
CAT_YOLO_WEIGHTS = MODELS_DIR / "cat" / "cat_yolo.pt"
HORSE_YOLO_WEIGHTS = MODELS_DIR / "horse" / "yolov8_horse_head.pt"

# --- Блок 1: Модерация ---
CLASSES = ["explicit", "safe", "violence"]
model = models.efficientnet_b0()
num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_ftrs, len(CLASSES))

if MODERATOR_WEIGHTS.exists():
    model.load_state_dict(torch.load(MODERATOR_WEIGHTS, map_location="cpu"))
    model.eval()
    print(f"✅ Moderator Model SUCCESS: Loaded from {MODERATOR_WEIGHTS}")
else:
    raise RuntimeError(f"Cannot start Moderator without weights: {MODERATOR_WEIGHTS}")

# --- Блок 2: Человеческая биометрия ---
try:
    face_app = FaceAnalysis(name="buffalo_s", root=str(WEIGHTS_DIR), providers=["CPUExecutionProvider"])
    face_app.prepare(ctx_id=-1, det_size=(640, 640))
    print("✅ Face Engine: Activated (Threshold: 0.4)")
except Exception as e:
    print(f"❌ Face Engine Error: {e}")

# --- Блок 3: Собачья биометрия ---
dog_service = None
try:
    from app.services.dog_biometrics_service import DogBiometricsService
    dog_service = DogBiometricsService(yolo_weights_path=str(DOG_YOLO_WEIGHTS))
    print("✅ Dog Biometrics Engine: Activated (DINOv2 ViT-S/14 384D)")
except Exception as e:
    print(f"❌ Dog Biometrics Engine Error: {e}")

# --- Блок 4: Кошачья биометрия ---
cat_service = None
try:
    from app.services.cat_biometrics_service import CatBiometricsService
    cat_service = CatBiometricsService(yolo_weights_path=str(CAT_YOLO_WEIGHTS))
    print("✅ Cat Biometrics Engine: Activated (DINOv2 ViT-S/14 384D)")
except Exception as e:
    print(f"❌ Cat Biometrics Engine Error: {e}")

# --- Блок 5: Лошадиная биометрия ---
horse_service = None
try:
    from app.services.horse_cascade_service import HorseCascadeService
    horse_service = HorseCascadeService(detector_path=str(HORSE_YOLO_WEIGHTS), top_k=3, match_threshold=14)
    print("✅ Horse Biometrics Engine: Activated (Tight Crop 15%, DINOv2 + LoFTR Th=14)")
except Exception as e:
    print(f"❌ Horse Biometrics Engine Error: {e}")

# --- Подключение ЕДИНОГО роутера биометрии ---
from app.api.v1.endpoints.biometrics import router as biometrics_router

app.include_router(biometrics_router, prefix="/api/v1", tags=["Biometrics"])

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

@app.post("/moderate")
async def moderate_image(file: UploadFile = File(...)):
    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        input_tensor = preprocess(image).unsqueeze(0)

        with torch.no_grad():
            features = model.features(input_tensor)
            pooled_features = model.avgpool(features)
            embedding = torch.flatten(pooled_features, 1).cpu().numpy().tolist()[0][:512]
            outputs = model.classifier(torch.flatten(pooled_features, 1))
            probs = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, class_idx = torch.max(probs, 0)

        verdict = CLASSES[class_idx.item()]
        conf_value = float(confidence.item())
        status = "allowed"
        if verdict != "safe":
            status = "blocked" if conf_value > 0.90 else "manual_review"

        return {
            "verdict": verdict,
            "status": status,
            "confidence": round(conf_value * 100, 2),
            "embedding": embedding,
            "filename": file.filename,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/recognize")
async def recognize_faces(file: UploadFile = File(...)):
    try:
        image_data = await file.read()
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Invalid image format")

        faces = face_app.get(img)
        results = [{
            "bbox": face.bbox.astype(int).tolist(),
            "conf": round(float(face.det_score), 4),
            "embedding": face.embedding.astype(float).tolist(),
        } for face in faces]

        return {"faces_found": len(results), "data": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, workers=1)