from pathlib import Path
from sklearn.model_selection import train_test_split

import pandas as pd


def load_meta(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.rename(columns={"path": "image"})
    df = df[["image", "age"]]
    return df


def filter_age_range(df: pd.DataFrame, min_age: int = 10, max_age: int = 75) -> pd.DataFrame:
    mask = (df["age"] > min_age) & (df["age"] < max_age)
    return df[mask].reset_index(drop=True)


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
