import torch.nn as nn
import torchvision.models as models


def create_resnet50_regressor(pretrained: bool = True) -> nn.Module:
    weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet50(weights=weights)

    for param in model.parameters():
        param.requires_grad = True  # или сначала заморозить часть слоёв

    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Linear(256, 1),
    )
    return model
