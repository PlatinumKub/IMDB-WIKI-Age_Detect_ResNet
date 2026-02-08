from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image
import imgaug.augmenters as iaa


def build_augmenter() -> iaa.Augmenter:
    return iaa.Sequential([
        iaa.Fliplr(0.5),
        iaa.Affine(rotate=(-30, 30)),
        iaa.Multiply((0.8, 1.2)),
        iaa.Crop(percent=(0, 0.1)),
        iaa.Sometimes(0.4, iaa.GaussianBlur(sigma=(0, 0.5))),
        iaa.Sometimes(0.4, iaa.AdditiveGaussianNoise(scale=(0, 0.05 * 255))),
        iaa.Sometimes(0.4, iaa.CoarseDropout(0.02, size_percent=(0.02, 0.05))),
        iaa.Sometimes(0.4, iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5))),
    ])


def augment_to_balance(
    df: pd.DataFrame,
    images_root: Path,
    output_subdir: str,
    class_column: str = "age",
    image_column: str = "image",
) -> pd.DataFrame:
    """
    df: DataFrame с колонками [image, age] (image - относительный путь типа imdb_crop/...).
    images_root: корень, относительно которого резолвим image (например, data/raw).
    output_subdir: имя подпапки, куда складывать аугментированные файлы (например, "augmented_train").
    """
    df = df.copy()
    classes: Iterable[int] = sorted(df[class_column].unique())
    target_count = df[class_column].value_counts().max()

    output_root = images_root / output_subdir
    output_root.mkdir(parents=True, exist_ok=True)

    augmenter = build_augmenter()
    new_rows = []

    for cls in classes:
        class_rows = df[df[class_column] == cls]
        class_images = class_rows[image_column].tolist()
        current_count = len(class_images)

        if current_count >= target_count:
            continue

        print(f"Augmenting class '{cls}' with {target_count - current_count} images.")
        generated = 0

        while current_count + generated < target_count:
            for rel_path in class_images:
                if current_count + generated >= target_count:
                    break

                src_path = images_root / rel_path
                image = Image.open(src_path).convert("RGB")
                image_np = np.array(image)

                augmented = augmenter(image=image_np)
                augmented_pil = Image.fromarray(augmented)

                aug_name = f"{cls}_{current_count + generated}.jpg"
                aug_rel_path = f"{output_subdir}/{aug_name}"
                aug_path = output_root / aug_name
                augmented_pil.save(aug_path)

                new_rows.append({image_column: aug_rel_path, class_column: cls})
                generated += 1

    if not new_rows:
        return df

    aug_df = pd.DataFrame(new_rows)
    balanced_df = pd.concat([df, aug_df], ignore_index=True)
    return balanced_df
