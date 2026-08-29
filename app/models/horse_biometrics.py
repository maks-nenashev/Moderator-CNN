import shutil
import tempfile
from pathlib import Path
from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from app.services.horse_cascade_service import HorseCascadeService

router = APIRouter(prefix="/horse", tags=["Horse Biometrics"])

# Сервис инициализируется единым синглтоном
cascade_service = HorseCascadeService(
    detector_path="/home/maks/Moderator-CNN/models/horse/yolov8_horse_head.pt",
    top_k=3,
    match_threshold=14
)


class SearchResponse(BaseModel):
    status: str
    matched_horse_id: str | None
    max_inliers: int
    threshold: int


@router.post("/enroll/{horse_id}")
async def enroll_horse_image(horse_id: str, file: UploadFile = File(...)):
    """Регистрация эталонного снимка лошади в галерею."""
    suffix = Path(file.filename).suffix or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        success = cascade_service.enroll_horse(horse_id, tmp_path)
        if not success:
            raise HTTPException(
                status_code=400, 
                detail="Голова лошади не обнаружена на снимке"
            )
        return {
            "status": "SUCCESS", 
            "horse_id": horse_id, 
            "filename": file.filename
        }
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@router.post("/search", response_model=SearchResponse)
async def search_horse(file: UploadFile = File(...)):
    """Каскадный поиск 1:N по базе особей."""
    suffix = Path(file.filename).suffix or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        result = cascade_service.search_1_to_n(tmp_path)
        if result.get("status") == "ERROR":
            raise HTTPException(status_code=422, detail=result.get("message"))
        return result
    finally:
        Path(tmp_path).unlink(missing_ok=True)