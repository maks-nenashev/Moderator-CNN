import io
import requests
from PIL import Image

SERVER_URL = "http://127.0.0.1:8001/api/v1/biometrics/dog/embedding"

def run_e2e_test():
    # 1. Генерация тестового изображения 300x300 в памяти
    img = Image.new("RGB", (300, 300), color=(128, 90, 40))
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="JPEG")
    img_bytes.seek(0)

    files = {"file": ("test_dog.jpg", img_bytes, "image/jpeg")}

    print(f"📡 Отправка запроса на {SERVER_URL}...")
    try:
        response = requests.post(SERVER_URL, files=files, timeout=10)
        
        print(f"Статус ответа: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Ответ сервера получен успешно!")
            print(f" Status: {data.get('status')}")
            print(f" Filename: {data.get('filename')}")
            print(f" Vector Dim: {data.get('vector_dim')}")
            print(f" First 5 float values: {data.get('embedding')[:5]}...")
            
            assert data.get("vector_dim") == 512, "Ошибка: Размерность вектора не равна 512!"
            assert len(data.get("embedding")) == 512, "Ошибка: Длина массива embedding не равна 512!"
            print("🎯 Валидация структуры ответа прошла успешно.")
        else:
            print(f"❌ Ошибка сервера: {response.text}")

    except requests.exceptions.ConnectionError:
        print("❌ Не удалось подключиться к серверу. Убедитесь, что Uvicorn запущен на порту 8001.")

if __name__ == "__main__":
    run_e2e_test()
