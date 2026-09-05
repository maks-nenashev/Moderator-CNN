import io
import logging
import shutil
import tempfile
from pathlib import Path
from fastapi import APIRouter, File, HTTPException, UploadFile, status
from PIL import Image

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
# HORSE EMBEDDING & SEARCH ENDPOINTS
# =====================================================================

@router.post("/horse/embedding")
@router.post("/biometrics/horse/embedding")
async def get_horse_embedding(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        logger.info(f"📥 [FASTAPI /horse/embedding] Processing: {file.filename}")

        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid image format: {e}"
            )

        service = get_horse_service()

        # Разрешение детектора из сервиса или фолбэк на YOLO
        detector = getattr(service, "detector", None)
        if detector is None and hasattr(service, "pipeline"):
            detector = getattr(service.pipeline, "detector", None)
        if detector is None:
            from ultralytics import YOLO
            from app.main import HORSE_YOLO_WEIGHTS
            detector = YOLO(str(HORSE_YOLO_WEIGHTS))

        # 1. Детекция головы лошади
        img_w, img_h = image.size
        results = detector(image, conf=0.35, verbose=False)
        boxes = results[0].boxes

        if len(boxes) == 0:
            logger.warning("⚠️ [FASTAPI HORSE] No horse head detected")
            return {
                "status": "no_horse_detected",
                "embedding": None,
                "bbox": None,
                "confidence": 0.0,
            }

        best_box = max(boxes, key=lambda b: float(b.conf[0]))
        conf = float(best_box.conf[0])
        x1, y1, x2, y2 = map(int, best_box.xyxy[0].tolist())

        # 2. Tight Crop (15% margin)
        w, h = x2 - x1, y2 - y1
        margin_w, margin_h = int(w * 0.15), int(h * 0.15)
        x1_c = max(0, x1 + margin_w)
        y1_c = max(0, y1 + margin_h)
        x2_c = min(img_w, x2 - margin_w)
        y2_c = min(img_h, y2 - margin_h)

        cropped_image = image.crop((x1_c, y1_c, x2_c, y2_c))

        # 3. Передача PIL.Image в DINOv2 Vector Service
        if hasattr(service, "vector_service"):
            vector_engine = service.vector_service
        elif hasattr(service, "pipeline") and hasattr(service.pipeline, "vector_service"):
            vector_engine = service.pipeline.vector_service
        else:
            raise AttributeError("Horse biometrics service missing vector_service instance")

# 3. Извлечение 384D эмбеддинга DINOv2
        raw_embedding = vector_engine.extract_embedding(cropped_image)

        # Безопасная приведение к единому float-списку (поддержка Tensor, ndarray, list)
        if hasattr(raw_embedding, "detach"):
            raw_embedding = raw_embedding.detach().cpu().numpy()

        if hasattr(raw_embedding, "flatten"):
            embedding = raw_embedding.flatten().tolist()
        elif isinstance(raw_embedding, (list, tuple)):
            embedding = [float(x) for x in raw_embedding]
        else:
            embedding = list(raw_embedding)

        logger.info(f"✅ [FASTAPI HORSE] Success | Conf: {round(conf, 4)}")

        return {
            "status": "success",
            "embedding": embedding,
            "bbox": {"x": x1_c, "y": y1_c, "w": x2_c - x1_c, "h": y2_c - y1_c},
            "confidence": round(conf, 4),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ [FASTAPI HORSE EMBEDDING] Exception: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Horse biometrics embedding engine failed: {str(e)}",
        )