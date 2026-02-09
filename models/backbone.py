import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights

class ModeratorCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(ModeratorCNN, self).__init__()
        # Загружаем предобученную базу
        self.backbone = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        # Меняем голову классификатора под наши 3 класса
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

def get_model(num_classes=3):
    return ModeratorCNN(num_classes=num_classes)
