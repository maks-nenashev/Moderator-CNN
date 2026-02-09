import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights

class ModeratorCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(ModeratorCNN, self).__init__()
        
        # 1. Загружаем backbone с предобученными весами
        # Weights.DEFAULT — это актуальный стандарт PyTorch для загрузки лучших весов
        self.backbone = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        
        # 2. Извлекаем количество входных признаков перед классификатором
        # У EfficientNet-B0 это 1280
        in_features = self.backbone.classifier[1].in_features
        
        # 3. Переопределяем голову (Classifier)
        # Мы заменяем стандартные 1000 классов ImageNet на наши нужды
        # Применяем Dropout для предотвращения переобучения
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

def get_model(num_classes=3):
    """
    Фабрика для создания модели. 
    num_classes: 0-Safe, 1-Explicit, 2-Violence (к примеру)
    """
    return ModeratorCNN(num_classes=num_classes)