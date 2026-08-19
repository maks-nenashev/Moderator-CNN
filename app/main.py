import io
from pathlib import Path
import cv2
from fastapi import FastAPI, File, HTTPException, UploadFile
from insightface.app import FaceAnalysis
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms

# 1. Создание экземпляра FastAPI
app = FastAPI(title="FindWay Master AI Engine", version="1.3.0")

# 2. Подключение роутеров (после инициализации app)
from app.api.v1.endpoints.biometrics import router as dog_biometrics_router

app.include_router(
    dog_biometrics_router, prefix="/api/v1", tags=["Dog Biometrics"]
)

# --- Настройки путей ---
BASE_DIR = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = BASE_DIR / "weights"

# --- Блок 1: Модерация (EfficientNet) ---
CLASSES = ["explicit", "safe", "violence"]
MODERATOR_WEIGHTS = WEIGHTS_DIR / "moderator_v1.pth"

model = models.efficientnet_b0()
num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_ftrs, len(CLASSES))

if MODERATOR_WEIGHTS.exists():
  model.load_state_dict(torch.load(MODERATOR_WEIGHTS, map_location="cpu"))
  model.eval()
  print("✅ Moderator: Activated")

# --- Блок 2: Человеческая биометрия (InsightFace) ---
try:
  face_app = FaceAnalysis(
      name="buffalo_s",
      root=str(WEIGHTS_DIR),
      providers=["CPUExecutionProvider"],
  )
  face_app.prepare(ctx_id=-1, det_size=(640, 640))
  print("✅ Face Engine: Activated (Threshold: 0.4)")
except Exception as e:
  print(f"❌ Face Engine Error: {e}")

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
      embedding = (
          torch.flatten(pooled_features, 1).cpu().numpy().tolist()[0][:512]
      )

      outputs = model.classifier(torch.flatten(pooled_features, 1))
      probs = torch.nn.functional.softmax(outputs[0], dim=0)
      confidence, class_idx = torch.max(probs, 0)

    verdict = CLASSES[class_idx]
    conf_value = float(confidence)

    status = "allowed"
    if verdict != "safe":
      if conf_value > 0.90:
        status = "blocked"
      elif conf_value > 0.60:
        status = "manual_review"

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

    faces = face_app.get(img)
    results = []
    for face in faces:
      results.append({
          "bbox": face.bbox.astype(int).tolist(),
          "conf": round(float(face.det_score), 4),
          "embedding": face.embedding.tolist(),
      })

    print(f"👤 [AI] {file.filename}: Found {len(results)} faces")
    return {"faces_found": len(results), "data": results}
  except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
  import uvicorn

  uvicorn.run(app, host="0.0.0.0", port=8001, workers=1)