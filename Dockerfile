FROM python:3.10-slim

# Системные зависимости
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libmagic1 \
    git \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    TORCH_HOME=/root/.cache/torch \
    YOLO_CONFIG_DIR=/tmp/Ultralytics

WORKDIR /app

# Установка зависимостей Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 1. Заморозка исходного кода репозитория DINOv2 на этапе сборки
RUN mkdir -p /root/.cache/torch/hub && \
    git clone https://github.com/facebookresearch/dinov2 /root/.cache/torch/hub/facebookresearch_dinov2_main

# 2. Копирование кода проекта
COPY . .

# 3. Принудительная заброска локальных кастомных весов в системный кэш
RUN mkdir -p /root/.cache/torch/hub/checkpoints && \
    if [ -d "weights/checkpoints" ]; then \
        cp -r weights/checkpoints/* /root/.cache/torch/hub/checkpoints/ ; \
    fi

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]