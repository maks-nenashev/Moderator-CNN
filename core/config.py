import logging
from pathlib import Path

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

# ------------------------------------------------------------------
# DOG BIOMETRICS CONFIG
# ------------------------------------------------------------------
DOG_YOLO_WEIGHTS: Path = MODELS_DIR / "dog" / "dog_yolo_dual.pt"
DOG_EMBEDDER_WEIGHTS: Path = MODELS_DIR / "dog" / "arcface_v1.pth"

# ------------------------------------------------------------------
# CAT BIOMETRICS CONFIG
# ------------------------------------------------------------------
CAT_YOLO_WEIGHTS: Path = MODELS_DIR / "cat" / "cat_yolo.pt"
CAT_EMBEDDER_WEIGHTS: Path = MODELS_DIR / "cat" / "arcface_v1.pth"