import torch.nn as nn
from efficientnet_pytorch import EfficientNet
import torchvision.models as models


def create_efficientnet_b2(num_outputs: int = 1, pretrained: bool = True) -> nn.Module:
    model = EfficientNet.from_pretrained("efficientnet-b2") if pretrained else EfficientNet.from_name("efficientnet-b2")
    model._fc = nn.Linear(model._fc.in_features, num_outputs)
    return model


def create_resnet18(num_outputs: int, pretrained: bool = True) -> nn.Module:
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, num_outputs)
    return model
