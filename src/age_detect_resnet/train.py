from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from age_detect_resnet.dataset import (
    create_dataloaders,
    CustomImageDataset,
    get_val_transforms,
)
from age_detect_resnet.model import create_resnet50_regressor
from age_detect_resnet.preprocessing.tabular import (
    load_meta,
    filter_age_range,
    limit_classes_to_n,
    split_df_train_val_test,
)


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_mae = 0.0
    num_batches = 0

    for inputs, labels in tqdm(train_loader, desc="Train", leave=False):
        inputs = inputs.to(device)
        labels = labels.float().view(-1, 1).to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        mae = torch.abs(outputs - labels).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_mae += mae.item()
        num_batches += 1

    return running_loss / num_batches, running_mae / num_batches


def validate_one_epoch(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_mae = 0.0
    num_batches = 0

    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Val", leave=False):
            inputs = inputs.to(device)
            labels = labels.float().view(-1, 1).to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            mae = torch.abs(outputs - labels).mean()

            running_loss += loss.item()
            running_mae += mae.item()
            num_batches += 1

    return running_loss / num_batches, running_mae / num_batches


def plot_curves(train_loss, val_loss, train_mae, val_mae):
    epochs = range(1, len(train_loss) + 1)
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label="Train")
    plt.plot(epochs, val_loss, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss vs. Epoch")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_mae, label="Train")
    plt.plot(epochs, val_mae, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.title("MAE vs. Epoch")
    plt.legend()

    plt.tight_layout()
    plt.show()


def evaluate_on_test(model, test_df, images_root, device, criterion):
    test_dataset = CustomImageDataset(
        test_df,
        images_root=images_root,
        transform=get_val_transforms(),
    )
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model.eval()
    running_mae = 0.0
    num_batches = 0

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Test", leave=False):
            inputs = inputs.to(device)
            labels = labels.float().view(-1, 1).to(device)

            outputs = model(inputs)
            mae = torch.abs(outputs - labels).mean()

            running_mae += mae.item()
            num_batches += 1

    test_mae = running_mae / num_batches
    print("MAE on test set:", test_mae)
    return test_mae


def main():
    torch.backends.cudnn.benchmark = True  # Авто-оптим conv kernels
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device() if torch.cuda.is_available() else 'CPU'}")
    print(f"GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
    DATA_RAW = PROJECT_ROOT / "data" / "raw"

    df = load_meta(DATA_PROCESSED / "meta.csv")
    df = filter_age_range(df)

    df = limit_classes_to_n(df, class_column="age", n=5000)
    train_df, val_df, test_df = split_df_train_val_test(df, stratify_col="age")

    train_loader, val_loader = create_dataloaders(
        train_df=train_df,
        val_df=val_df,
        images_root=DATA_RAW,
        batch_size=128,
        num_workers=4,
    )

    model = create_resnet50_regressor(pretrained=True).to(device)
    criterion = nn.MSELoss()
    optimizer = AdamW(model.parameters(), lr=5e-5)

    NUM_EPOCHS = 20
    best_mae = float("inf")
    train_loss_hist = []
    val_loss_hist = []
    train_mae_hist = []
    val_mae_hist = []

    for epoch in range(NUM_EPOCHS):
        train_loss, train_mae = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_mae = validate_one_epoch(
            model, val_loader, criterion, device
        )

        train_loss_hist.append(train_loss)
        val_loss_hist.append(val_loss)
        train_mae_hist.append(train_mae)
        val_mae_hist.append(val_mae)

        print(
            f"Epoch {epoch+1}/{NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.6f}, Train MAE: {train_mae:.6f} | "
            f"Val Loss: {val_loss:.6f}, Val MAE: {val_mae:.6f}"
        )

        if val_mae < best_mae:
            best_mae = val_mae
            torch.save(model.state_dict(), PROJECT_ROOT / "best_model.pth")
            print("Saved best model!")

        if epoch > 1 and abs(val_loss_hist[-1] - val_loss_hist[-2]) < 1e-5:
            print("Validation loss converges")
            break

    plot_curves(train_loss_hist, val_loss_hist, train_mae_hist, val_mae_hist)

    test_df = test_df[(test_df["age"] < 60) & (test_df["age"] > 15)]
    best_model = create_resnet50_regressor(pretrained=False).to(device)
    best_model.load_state_dict(torch.load(PROJECT_ROOT / "best_model.pth", map_location=device))

    evaluate_on_test(best_model, test_df, images_root=DATA_RAW, device=device, criterion=criterion)


if __name__ == "__main__":
    main()
