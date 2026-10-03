import os
import torch

def load_local_dinov2(model_name: str = "dinov2_vits14", device: str = "cpu"):
    hub_dir = os.path.expanduser("~/.cache/torch/hub/facebookresearch_dinov2_main")
    
    if not os.path.exists(hub_dir):
        raise FileNotFoundError(f"Локальный кэш репозитория DINOv2 не найден в {hub_dir}")

    # 1. Загрузка архитектуры модели из локального hubconf.py без выхода в сеть
    model = torch.hub.load(hub_dir, model_name, source="local", pretrained=False)

    # 2. Поиск локального файла кастомных весов
    possible_paths = [
        "/app/weights/checkpoints/dinov2_vits14_pretrain.pth",
        "./weights/checkpoints/dinov2_vits14_pretrain.pth"
    ]
    weights_path = next((p for p in possible_paths if os.path.exists(p)), None)

    if weights_path:
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        raise FileNotFoundError("Не найден файл весов dinov2_vits14_pretrain.pth")

    return model.to(device).eval()