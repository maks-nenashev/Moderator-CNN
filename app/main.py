from fastapi import FastAPI, UploadFile, File, HTTPException
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import numpy as np
from pathlib import Path

app = FastAPI(title="FindWay Moderator & Vector Engine", version="1.1.0")

# --- Ініціалізація моделі ---
CLASSES = ['explicit', 'safe', 'violence']
MODEL_PATH = "weights/moderator_v1.pth"

model = models.efficientnet_b0()
num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_ftrs, len(CLASSES))

if Path(MODEL_PATH).exists():
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()
    print(f"✅ Модель з векторами активована")

# Функція-хук для витягування вектора (Embeddings)
# Ми беремо дані з шару перед класифікатором
embeddings = []
def hook(module, input, output):
    embeddings.append(output.detach().flatten(1).numpy().tolist()[0])

model.avgpool.register_forward_hook(hook)

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.post("/moderate")
async def moderate_image(file: UploadFile = File(...)):
    global embeddings
    embeddings = [] # Очищуємо перед кожним запитом
    
    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        input_tensor = preprocess(image).unsqueeze(0)

        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, class_idx = torch.max(probs, 0)

        # Формуємо відповідь з вектором для Rails
        return {
            "verdict": CLASSES[class_idx],
            "confidence": round(float(confidence) * 100, 2),
            "embedding": embeddings[0] if embeddings else [], # ОСЬ ВІН - ЦИФРОВИЙ ВІДБИТОК
            "filename": file.filename
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
