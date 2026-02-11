from pathlib import Path

import torch
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from age_detect_resnet.model import create_resnet50_regressor
from age_detect_resnet.preprocessing.tabular import (
    load_meta,
    filter_age_range,
    limit_classes_to_n,
    split_df_train_val_test,
)


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


def visualize_predictions_grid(
    model,
    df,
    device: torch.device,
    images_root: Path,
    num_samples: int = 6,
):
    model.eval()
    preprocess = get_preprocess()

    sample_df = df.sample(num_samples, random_state=42).reset_index(drop=True)

    rows, cols = 2, 3
    fig, axs = plt.subplots(rows, cols, figsize=(12, 8))
    axs = axs.ravel()

    max_cells = rows * cols
    num_show = min(num_samples, max_cells)

    for i in range(num_show):
        row = sample_df.iloc[i]
        img_path = images_root / row["image"]
        true_age = row["age"]

        image = Image.open(img_path).convert("RGB")
        tensor = preprocess(image).unsqueeze(0).to(device)

        with torch.no_grad():
            pred_age = model(tensor).item()

        axs[i].imshow(image)
        axs[i].set_title(f"True: {true_age}\nPred: {pred_age:.0f}", fontsize=10)
        axs[i].axis("off")

    # выключить лишние оси, если num_samples < rows*cols
    for j in range(num_show, max_cells):
        axs[j].axis("off")

    plt.tight_layout()
    plt.show()


def main():
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    DATA_RAW = PROJECT_ROOT / "data" / "raw"
    DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_trained_model(PROJECT_ROOT / "best_model.pth", device)

    # загружаем тот же meta.csv и делаем такие же препроц шаги, как в train.py,
    # чтобы test_df совпадал по распределению
    df = load_meta(DATA_PROCESSED / "meta.csv")
    df = filter_age_range(df)
    df = limit_classes_to_n(df, class_column="age", n=5000)
    _, _, test_df = split_df_train_val_test(df, stratify_col="age")

    # можно фильтрануть возраст, как в evaluate_on_test:
    test_df = test_df[(test_df["age"] < 60) & (test_df["age"] > 15)]

    visualize_predictions_grid(
        model=model,
        df=test_df,
        device=device,
        images_root=DATA_RAW,
        num_samples=12,
    )

if __name__ == "__main__":
    main()
