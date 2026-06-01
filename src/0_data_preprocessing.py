from pathlib import Path

import pandas as pd
from sklearn.impute import KNNImputer
import warnings

from feature_schema import normalize_pm25_column, sort_by_time
from config import DATA_DIR, RAW_DIR


warnings.filterwarnings('ignore')

TIME_COLUMNS = ["year", "month", "day", "hour"]


def _load_raw_frames():
    raw_files = sorted(RAW_DIR.glob("*.csv"))
    if not raw_files:
        raise FileNotFoundError(f"Không tìm thấy file CSV nào trong {RAW_DIR}")

    frames = [pd.read_csv(path) for path in raw_files]
    return pd.concat(frames, axis=0, ignore_index=True)


def _impute_numeric_columns(df):
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    numeric_cols = [column for column in numeric_cols if column not in TIME_COLUMNS]
    if numeric_cols:
        df[numeric_cols] = KNNImputer(n_neighbors=5).fit_transform(df[numeric_cols])
    return df


def preprocess_raw_data():
    print("🛠️ [PREPROCESSING] Đang gộp file raw và làm sạch dữ liệu...")
    df = normalize_pm25_column(_load_raw_frames())

    if "PM2_5" in df.columns:
        q1, q3 = df["PM2_5"].quantile(0.25), df["PM2_5"].quantile(0.75)
        iqr = q3 - q1
        df = df.loc[~((df["PM2_5"] < q1 - 1.5 * iqr) | (df["PM2_5"] > q3 + 1.5 * iqr))]

    df = sort_by_time(df)

    if "wd" in df.columns:
        df["wd"] = df["wd"].fillna(df["wd"].mode(dropna=True)[0])

    if "station" in df.columns:
        df["station"] = df["station"].astype("string").str.strip()
    if "wd" in df.columns:
        df["wd"] = df["wd"].astype("string").str.strip()

    df = _impute_numeric_columns(df)

    output_path = DATA_DIR / "airquality_data.csv"
    df.to_csv(output_path, index=False)
    print(f"✅ [PREPROCESSING] Đã lưu file sạch tại {output_path}")

if __name__ == "__main__":
    preprocess_raw_data()