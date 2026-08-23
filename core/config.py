import logging
from pathlib import Path

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

# Точные имена из вашего дерева каталогов:
DOG_YOLO_WEIGHTS = MODELS_DIR / "dog" / "dog_yolo_dual.pt"
#DOG_EMBEDDER_WEIGHTS = MODELS_DIR / "dog" / "moderator_v1.pth"
DOG_EMBEDDER_WEIGHTS: Path = MODELS_DIR / "dog" / "arcface_v1.pth"