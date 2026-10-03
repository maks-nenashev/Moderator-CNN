import io
import requests
from PIL import Image

BASE = "http://127.0.0.1:8001"

def test_everything():
    img = Image.new("RGB", (300, 300), color=(120, 120, 120))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")

    endpoints = [
        ("/moderate", "Модерация (EfficientNet)"),
        ("/recognize", "Человеческая биометрия (InsightFace)"),
        ("/api/v1/biometrics/dog/embedding", "Собачья биометрия (DogBiometrics)")
    ]

    print("🔍 Комплексная проверка всех модулей Moderator-CNN:\n")
    for path, name in endpoints:
        buf.seek(0)
        r = requests.post(f"{BASE}{path}", files={"file": ("test.jpg", buf, "image/jpeg")})
        status = "✅ OK" if r.status_code == 200 else f"❌ Error {r.status_code}"
        print(f"  • {name} [{path}]: {status}")

if __name__ == "__main__":
    test_everything()
