import sys
from pathlib import Path

# Добавление корня репозитория в sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
from models.dog_biometrics import DogBiometricNet

def test_embedding_shape_and_norm():
    model = DogBiometricNet(pretrained=False)
    model.eval()
    
    dummy_input = torch.randn(1, 3, 224, 224)
    
    with torch.no_grad():
        embedding = model(dummy_input)
    
    assert embedding.shape == (1, 512), f"Неверная размерность: {embedding.shape}"
    
    norm = torch.norm(embedding, p=2, dim=1).item()
    assert abs(norm - 1.0) < 1e-5, f"Ошибка L2-нормализации: norm={norm}"
    
    print("✅ Биометрический вектор собачьего профиля успешно получен!")
    print(f"Размерность: {list(embedding.shape)}")
    print(f"L2-норма: {norm:.6f}")

if __name__ == "__main__":
    test_embedding_shape_and_norm()
