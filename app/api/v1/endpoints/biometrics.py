import logging
from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.services.cat_biometrics_service import CatBiometricsService
from app.services.dog_biometrics_service import DogBiometricsService

logger = logging.getLogger(__name__)

router = APIRouter()

_dog_biometrics_service: DogBiometricsService | None = None
_cat_biometrics_service: CatBiometricsService | None = None


def get_dog_service() -> DogBiometricsService:
    """
    Синглтон-провайдер для собак: приоритетно переиспользует dog_service из app.main,
    избегая повторной загрузки весов в VRAM/RAM.
    """
    global _dog_biometrics_service
    if _dog_biometrics_service is not None:
        return _dog_biometrics_service

    # 1. Попытка переиспользовать уже инициализированный экземпляр из app.main
    try:
        from app.main import dog_service
        if dog_service is not None:
            _dog_biometrics_service = dog_service
            return _dog_biometrics_service
    except ImportError:
        logger.warning("⚠️ Не удалось импортировать dog_service из app.main, переход к фолбэку")

    # 2. Фолбэк: ленивая инициализация из конфигурации
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
    """
    Синглтон-провайдер для кошек: приоритетно переиспользует cat_service из app.main,
    избегая повторной загрузки весов в VRAM/RAM.
    """
    global _cat_biometrics_service
    if _cat_biometrics_service is not None:
        return _cat_biometrics_service

    # 1. Попытка переиспользовать уже инициализированный экземпляр из app.main
    try:
        from app.main import cat_service
        if cat_service is not None:
            _cat_biometrics_service = cat_service
            return _cat_biometrics_service
    except ImportError:
        logger.warning("⚠️ Не удалось импортировать cat_service из app.main, переход к фолбэку")

    # 2. Фолбэк: ленивая инициализация из конфигурации
    try:
        from app.core.config import CAT_EMBEDDER_WEIGHTS, CAT_YOLO_WEIGHTS
        _cat_biometrics_service = CatBiometricsService(
            embedder_weights_path=str(CAT_EMBEDDER_WEIGHTS),
            yolo_weights_path=str(CAT_YOLO_WEIGHTS),
        )
        return _cat_biometrics_service
    except Exception:
        # Фолбэк на дефолтную инициализацию CatBiometricsService, если веса в config не объявлены
        _cat_biometrics_service = CatBiometricsService()
        return _cat_biometrics_service


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