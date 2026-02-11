from pathlib import Path

import torch
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from age_detect_resnet.model import create_resnet50_regressor


def get_preprocess():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def load_trained_model(checkpoint_path: str | Path, device: torch.device):
    model = create_resnet50_regressor(pretrained=False)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def predict_age_for_image(image_path: str | Path, model, device: torch.device):
    preprocess = get_preprocess()
    image = Image.open(image_path).convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)  # shape [1, 1]
    age_pred = output.item()
    return age_pred, image


def demo_images(image_paths):
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_trained_model(PROJECT_ROOT / "best_model.pth", device)

    for p in image_paths:
        img_path = PROJECT_ROOT / p
        age_pred, image = predict_age_for_image(img_path, model, device)
        plt.figure()
        plt.title(f"Predict: {int(age_pred)}")
        plt.axis("off")
        plt.imshow(image)
        plt.show()


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    demo_images([PROJECT_ROOT / "data" / "me.jpg"])
