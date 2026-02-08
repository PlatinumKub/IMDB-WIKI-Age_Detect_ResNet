import numpy as np
from scipy.io import loadmat
import pandas as pd
import datetime as date
from dateutil.relativedelta import relativedelta
from pathlib import Path


COLS = ["age", "gender", "path", "face_score1", "face_score2"]


def _build_paths(full_paths, prefix: str):
    return [f"{prefix}/{p[0]}" for p in full_paths]


def _decode_genders(raw_gender):
    genders = []
    for g in raw_gender:
        genders.append("male" if g == 1 else "female")
    return genders


def _compute_ages(dob_list, years_list):
    ages = []
    for i in range(len(dob_list)):
        try:
            d1 = date.datetime.strptime(dob_list[i][0:10], "%Y-%m-%d")
            d2 = date.datetime.strptime(str(years_list[i]), "%Y")
            diff = relativedelta(d2, d1).years
        except Exception:
            diff = -1
        ages.append(diff)
    return ages


def build_meta(imdb_mat_path: str,
               wiki_mat_path: str,
               output_csv: str = "meta.csv") -> None:
    imdb_data = loadmat(imdb_mat_path)
    wiki_data = loadmat(wiki_mat_path)

    imdb = imdb_data["imdb"]
    wiki = wiki_data["wiki"]

    imdb_photo_taken = imdb[0][0][1][0]
    imdb_full_path = imdb[0][0][2][0]
    imdb_gender = imdb[0][0][3][0]
    imdb_face_score1 = imdb[0][0][6][0]
    imdb_face_score2 = imdb[0][0][7][0]

    wiki_photo_taken = wiki[0][0][1][0]
    wiki_full_path = wiki[0][0][2][0]
    wiki_gender = wiki[0][0][3][0]
    wiki_face_score1 = wiki[0][0][6][0]
    wiki_face_score2 = wiki[0][0][7][0]

    imdb_path = _build_paths(imdb_full_path, "imdb_crop")
    wiki_path = _build_paths(wiki_full_path, "wiki_crop")

    imdb_genders = _decode_genders(imdb_gender)
    wiki_genders = _decode_genders(wiki_gender)

    imdb_dob = []
    for file in imdb_path:
        temp = file.split("_")[3].split("-")
        if len(temp[1]) == 1:
            temp[1] = "0" + temp[1]
        if len(temp[2]) == 1:
            temp[2] = "0" + temp[2]

        if temp[1] == "00":
            temp[1] = "01"
        if temp[2] == "00":
            temp[2] = "01"

        imdb_dob.append("-".join(temp))

    wiki_dob = [file.split("_")[2] for file in wiki_path]

    imdb_age = _compute_ages(imdb_dob, imdb_photo_taken)
    wiki_age = _compute_ages(wiki_dob, wiki_photo_taken)

    final_imdb = np.vstack(
        (imdb_age, imdb_genders, imdb_path, imdb_face_score1, imdb_face_score2)
    ).T
    final_wiki = np.vstack(
        (wiki_age, wiki_genders, wiki_path, wiki_face_score1, wiki_face_score2)
    ).T

    final_imdb_df = pd.DataFrame(final_imdb, columns=COLS)
    final_wiki_df = pd.DataFrame(final_wiki, columns=COLS)

    meta = pd.concat((final_imdb_df, final_wiki_df))

    meta = meta[meta["face_score1"] != "-inf"]
    meta = meta[meta["face_score2"] == "nan"]

    meta = meta.drop(["face_score1", "face_score2"], axis=1)

    meta = meta.sample(frac=1).reset_index(drop=True)

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    meta.to_csv(output_csv, index=False)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / 'data'

if __name__ == "__main__":
    build_meta(
        DATA_DIR / "raw" / "imdb_crop" / "imdb.mat",
        DATA_DIR / "raw" / "wiki_crop" / "wiki.mat",
        DATA_DIR / "processed" / "meta.csv",
    )
