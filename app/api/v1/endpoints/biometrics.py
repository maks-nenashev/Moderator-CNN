import logging
import shutil
import tempfile
from pathlib import Path
from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.services.cat_biometrics_service import CatBiometricsService
from app.services.dog_biometrics_service import DogBiometricsService
from app.services.horse_cascade_service import HorseCascadeService

logger = logging.getLogger(__name__)

router = APIRouter()

_dog_biometrics_service: DogBiometricsService | None = None
_cat_biometrics_service: CatBiometricsService | None = None
_horse_biometrics_service: HorseCascadeService | None = None


def get_dog_service() -> DogBiometricsService:
    global _dog_biometrics_service
    if _dog_biometrics_service is not None:
        return _dog_biometrics_service

    try:
        from app.main import dog_service
        if dog_service is not None:
            _dog_biometrics_service = dog_service
            return _dog_biometrics_service
    except ImportError:
        logger.warning("⚠️ Не удалось импортировать dog_service из app.main, переход к фолбэку")

    try:
        from app.core.config import DOG_EMBEDDER_WEIGHTS, DOG_YOLO_WEIGHTS
        _dog_biometrics_service = DogBiometricsService(
            embedder_weights_path=str(DOG_EMBEDDER_WEIGHTS),
            yolo_weights_path=str(DOG_YOLO_WEIGHTS),
        )
        return _dog_biometrics_service
    except Exception as e:
        logger.error(f"❌ Критическая ошибка создания DogBiometricsService: {e}")
        raise RuntimeError(f"Failed to initialize DogBiometricsService: {e}")


def get_cat_service() -> CatBiometricsService:
    global _cat_biometrics_service
    if _cat_biometrics_service is not None:
        return _cat_biometrics_service

    try:
        from app.main import cat_service
        if cat_service is not None:
            _cat_biometrics_service = cat_service
            return _cat_biometrics_service
    except ImportError:
        logger.warning("⚠️ Не удалось импортировать cat_service из app.main, переход к фолбэку")

    try:
        from app.core.config import CAT_EMBEDDER_WEIGHTS, CAT_YOLO_WEIGHTS
        _cat_biometrics_service = CatBiometricsService(
            embedder_weights_path=str(CAT_EMBEDDER_WEIGHTS),
            yolo_weights_path=str(CAT_YOLO_WEIGHTS),
        )
        return _cat_biometrics_service
    except Exception:
        _cat_biometrics_service = CatBiometricsService()
        return _cat_biometrics_service


def get_horse_service() -> HorseCascadeService:
    """
    Синглтон-провайдер для лошадей: переиспользует horse_service из app.main.
    """
    global _horse_biometrics_service
    if _horse_biometrics_service is not None:
        return _horse_biometrics_service

    try:
        from app.main import horse_service
        if horse_service is not None:
            _horse_biometrics_service = horse_service
            return _horse_biometrics_service
    except ImportError:
        logger.warning("⚠️ Не удалось импортировать horse_service из app.main, переход к фолбэку")

    try:
        from app.main import HORSE_YOLO_WEIGHTS
        _horse_biometrics_service = HorseCascadeService(
            detector_path=str(HORSE_YOLO_WEIGHTS),
            top_k=3,
            match_threshold=14
        )
        return _horse_biometrics_service
    except Exception as e:
        logger.error(f"❌ Критическая ошибка создания HorseCascadeService: {e}")
        raise RuntimeError(f"Failed to initialize HorseCascadeService: {e}")


# =====================================================================
# DOG EMBEDDING ENDPOINTS
# =====================================================================

@router.post("/dog/embedding")
@router.post("/biometrics/dog/embedding")
async def get_dog_embedding(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        logger.info(f"📥 [FASTAPI /dog/embedding] Processing: {file.filename}")

        service = get_dog_service()
        result = service.process_image(image_bytes)

        status_code = result.get("status", "error")

        if status_code != "success":
            logger.warning(f"⚠️ [FASTAPI DOG] Detector rejected: {status_code}")
            return {
                "status": status_code,
                "embedding": None,
                "bbox": None,
                "confidence": result.get("confidence", 0.0),
            }

        logger.info(
            f"✅ [FASTAPI DOG] Success | BBox: {result['bbox']} | Conf: {result['confidence']}"
        )

        return {
            "status": "success",
            "embedding": result["embedding"],
            "bbox": result["bbox"],
            "confidence": result["confidence"],
        }

    except Exception as e:
        logger.error(f"❌ [FASTAPI DOG] Exception: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Dog biometrics engine failed: {str(e)}",
        )


# =====================================================================
# CAT EMBEDDING ENDPOINTS
# =====================================================================

@router.post("/cat/embedding")
@router.post("/biometrics/cat/embedding")
async def get_cat_embedding(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        logger.info(f"📥 [FASTAPI /cat/embedding] Processing: {file.filename}")

        service = get_cat_service()
        result = service.process_image(image_bytes)

        status_code = result.get("status", "error")

        if status_code != "success":
            logger.warning(f"⚠️ [FASTAPI CAT] Detector rejected: {status_code}")
            return {
                "status": status_code,
                "embedding": None,
                "bbox": None,
                "confidence": result.get("confidence", 0.0),
            }

        logger.info(
            f"✅ [FASTAPI CAT] Success | BBox: {result['bbox']} | Conf: {result['confidence']}"
        )

        return {
            "status": "success",
            "embedding": result["embedding"],
            "bbox": result["bbox"],
            "confidence": result["confidence"],
        }

    except Exception as e:
        logger.error(f"❌ [FASTAPI CAT] Exception: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Cat biometrics engine failed: {str(e)}",
        )


# =====================================================================
# HORSE BIOMETRICS ENDPOINTS
# =====================================================================

@router.post("/horse/enroll/{horse_id}")
@router.post("/biometrics/horse/enroll/{horse_id}")
async def enroll_horse_image(horse_id: str, file: UploadFile = File(...)):
    try:
        service = get_horse_service()
        suffix = Path(file.filename).suffix or ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        try:
            success = service.enroll_horse(horse_id, tmp_path)
            if not success:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Голова лошади не обнаружена на снимке"
                )
            return {"status": "SUCCESS", "horse_id": horse_id, "filename": file.filename}
        finally:
            Path(tmp_path).unlink(missing_ok=True)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ [FASTAPI HORSE ENROLL] Exception: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Horse enrollment failed: {str(e)}",
        )


@router.post("/horse/search")
@router.post("/biometrics/horse/search")
async def search_horse(file: UploadFile = File(...)):
    try:
        service = get_horse_service()
        suffix = Path(file.filename).suffix or ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        try:
            result = service.search_1_to_n(tmp_path)
            if result.get("status") == "ERROR":
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=result.get("message")
                )
            return result
        finally:
            Path(tmp_path).unlink(missing_ok=True)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ [FASTAPI HORSE SEARCH] Exception: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Horse biometrics search failed: {str(e)}",
        )