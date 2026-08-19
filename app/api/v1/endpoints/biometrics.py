from pathlib import Path
from typing import Optional
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from app.services.dog_biometrics_service import DogBiometricsService

router = APIRouter()

BASE_DIR = Path(__file__).resolve().parents[4]
WEIGHTS_PATH = BASE_DIR / "weights" / "dog_biometrics_efficientnet_b0.pth"

dog_service = DogBiometricsService(weights_path=str(WEIGHTS_PATH))


@router.post("/biometrics/dog/embedding")
async def get_dog_embedding(
    file: UploadFile = File(...),
    x: Optional[int] = Form(0),
    y: Optional[int] = Form(0),
    w: Optional[int] = Form(0),
    h: Optional[int] = Form(0),
):
  contents = await file.read()
  if not contents:
    raise HTTPException(status_code=400, detail="Empty file provided")

  try:
    bbox_input = [x, y, w, h] if (w > 0 and h > 0) else None

    # Получение L2-нормализованного вектора
    result = dog_service.predict_embedding(contents, bbox=bbox_input)

    return {
        "status": "success",
        "bounding_box": {
            "x": result["bbox"][0],
            "y": result["bbox"][1],
            "w": result["bbox"][2],
            "h": result["bbox"][3],
        },
        "embedding": result["embedding"],
        "embedding_size": len(result["embedding"]),
    }
  except Exception as e:
    raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")