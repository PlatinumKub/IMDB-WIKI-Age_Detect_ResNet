from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split


DEFAULT_BINS = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75]

DEFAULT_AGE_RANGE_MAPPING: Dict[str, int] = {
    "10-14": 0,
    "15-19": 1,
    "20-24": 2,
    "25-29": 3,
    "30-34": 4,
    "35-39": 5,
    "40-44": 6,
    "45-49": 7,
    "50-54": 8,
    "55-59": 9,
    "60-64": 10,
    "65-69": 11,
    "70-74": 12,
}


def load_meta(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.rename(columns={"path": "image"})
    df = df[["image", "age"]]
    return df


def filter_age_range(df: pd.DataFrame, min_age: int = 10, max_age: int = 75) -> pd.DataFrame:
    mask = (df["age"] > min_age) & (df["age"] < max_age)
    return df[mask].reset_index(drop=True)


def age_to_range(age: int, bins: List[int]) -> str | None:
    if not isinstance(age, (int, np.integer)) or age < 0:
        raise ValueError("Age must be a non-negative integer.")

    index = np.digitize(age, bins) - 1

    if 0 <= index < len(bins) - 1:
        return f"{bins[index]}-{bins[index + 1] - 1}"
    else:
        return None


def add_age_range_column(
    df: pd.DataFrame,
    age_column: str = "age",
    bins: List[int] | None = None,
    new_column: str = "age_range",
) -> pd.DataFrame:
    if bins is None:
        bins = DEFAULT_BINS

    if age_column not in df.columns:
        raise ValueError(f"Column '{age_column}' not found in DataFrame.")

    df = df.copy()
    df[new_column] = df[age_column].apply(lambda age: age_to_range(age, bins))
    return df


def encode_age_ranges(
    df: pd.DataFrame,
    mapping: Dict[str, int] | None = None,
    source_column: str = "age_range",
    target_column: str = "age",
) -> pd.DataFrame:
    if mapping is None:
        mapping = DEFAULT_AGE_RANGE_MAPPING

    if source_column not in df.columns:
        raise ValueError(f"Column '{source_column}' not found in DataFrame.")

    df = df.copy()
    df[target_column] = df[source_column].map(mapping)
    if df[target_column].isna().any():
        raise ValueError("Some age ranges could not be mapped to integer classes.")
    df[target_column] = df[target_column].astype(int)
    return df[["image", target_column]]


def limit_classes_to_n(df: pd.DataFrame, class_column: str = "age", n: int = 5000) -> pd.DataFrame:
    limited_df_list = []
    for cls in sorted(df[class_column].unique()):
        class_rows = df[df[class_column] == cls]
        limited_df_list.append(class_rows.head(n))
    limited_df = pd.concat(limited_df_list, ignore_index=True)
    return limited_df


def split_df_train_val_test(
    df: pd.DataFrame,
    test_size: float = 0.1,
    val_size: float = 0.3,
    stratify_col: str | None = "age",
    random_state: int = 42,
):
    stratify = df[stratify_col] if stratify_col is not None else None
    train_val, test = train_test_split(
        df, test_size=test_size, stratify=stratify, random_state=random_state
    )

    stratify_tv = train_val[stratify_col] if stratify_col is not None else None
    train, val = train_test_split(
        train_val, test_size=val_size, stratify=stratify_tv, random_state=random_state
    )
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)
