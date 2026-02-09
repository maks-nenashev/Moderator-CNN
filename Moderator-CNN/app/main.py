cat << 'EOF' > app/main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
import shutil
import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
from pathlib import Path

app = FastAPI(title="FindWay Moderator CNN", version="1.0.0")

# --- Инициализация модели (Stability & Risk Control) ---
CLASSES = ['explicit', 'safe', 'violence']
MODEL_PATH = "weights/moderator_v1.pth"

# Загружаем архитектуру EfficientNet-B0
model = models.efficientnet_b0()
num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_ftrs, len(CLASSES))

# Загружаем веса
if Path(MODEL_PATH).exists():
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()
    print(f"✅ Модель успешно загружена из {MODEL_PATH}")
else:
    print(f"⚠️ ВНИМАНИЕ: Файл весов {MODEL_PATH} не найден! Модель будет выдавать случайные результаты.")

# Препроцессинг (Reproducibility)
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.get("/health")
def health_check():
    return {
        "status": "online", 
        "engine": "PyTorch", 
        "backbone": "EfficientNet-B0",
        "weights_found": Path(MODEL_PATH).exists()
    }

@app.post("/moderate")
async def moderate_image(file: UploadFile = File(...)):
    # 1. Валидация контента
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        # 2. Чтение изображения напрямую в память (без TMP_DIR для скорости)
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')

        # 3. Инференс
        input_tensor = preprocess(image).unsqueeze(0)
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, class_idx = torch.max(probs, 0)

        # 4. Логика вердикта с учетом порога уверенности (Risk Control)
        verdict = CLASSES[class_idx]
        conf_value = float(confidence)
        
        # Настройка "Умного порога"
        status = "allowed"
        if verdict != "safe":
            if conf_value > 0.90:
                status = "blocked" # Высокая уверенность в нарушении
            elif conf_value > 0.60:
                status = "manual_review" # Сомнительно
            else:
                status = "allowed" # Слишком низкая уверенность, пропускаем

        return {
            "filename": file.filename,
            "verdict": verdict,
            "confidence": round(conf_value * 100, 2),
            "status": status,
            "predictions": {CLASSES[i]: round(float(probs[i]), 4) for i in range(len(probs))}
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Запускаем на порту 8000 (стандарт для FastAPI)
    uvicorn.run(app, host="0.0.0.0", port=8000)
EOF