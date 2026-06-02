import json
from collections import defaultdict, deque
from pathlib import Path

import lightgbm as lgb
import pandas as pd
import xgboost as xgb

from config import MODEL_LIGHTGBM_PATH, MODEL_XGBOOST_PATH
from feature_schema import add_physical_features, normalize_pm25_column, prepare_inference_frame


class RealtimePredictor:
    def __init__(self):
        self.model_name = ""
        self.features = []
        self.target_col = "Target_PM2.5_next_1h"
        self._model = None
        # Lưu 2 dòng gần nhất theo station để tính lag/rolling realtime.
        self._history_by_station: dict[str, deque[dict]] = defaultdict(lambda: deque(maxlen=2))
        # Lưu running sum cho cum_wspm theo station.
        self._cum_wspm_by_station: dict[str, float] = defaultdict(float)
        self._load_model()

    def _candidate_metadata_files(self):
        root = Path(__file__).resolve().parent
        return [
            root / "metadata_lightgbm.json",
            root / "metadata_xgboost.json",
            root / "model_metadata.json",
        ]

    def _load_metadata(self):
        for path in self._candidate_metadata_files():
            if path.exists():
                data = json.loads(path.read_text(encoding="utf-8"))
                if data.get("model_name") and data.get("feature_columns"):
                    return data
        raise FileNotFoundError(
            "Không tìm thấy metadata model. Hãy chạy train trước (3_train_model.py)."
        )

    def _load_model(self):
        metadata = self._load_metadata()
        self.model_name = str(metadata["model_name"]).lower()
        self.features = list(metadata["feature_columns"])
        self.target_col = metadata.get("target_col", self.target_col)

        if self.model_name == "lightgbm":
            if not MODEL_LIGHTGBM_PATH.exists():
                raise FileNotFoundError(f"Thiếu model LightGBM: {MODEL_LIGHTGBM_PATH}")
            self._model = lgb.Booster(model_file=str(MODEL_LIGHTGBM_PATH))
            return

        if self.model_name == "xgboost":
            if not MODEL_XGBOOST_PATH.exists():
                raise FileNotFoundError(f"Thiếu model XGBoost: {MODEL_XGBOOST_PATH}")
            model = xgb.Booster()
            model.load_model(str(MODEL_XGBOOST_PATH))
            self._model = model
            return

        raise ValueError(f"Model chưa hỗ trợ realtime: {self.model_name}")

    @staticmethod
    def _safe_float(value, default=0.0) -> float:
        try:
            if value is None:
                return float(default)
            return float(value)
        except Exception:
            return float(default)

    def _enrich_one_event(self, row: dict) -> dict:
        """
        Enrich 1 event với:
        - physical features (hour_sin/cos, month_sin/cos, vapor pressure)
        - lag_1, lag_2 cho các numeric columns có trong row
        - rolling_2 cho các numeric columns (mean của [prev, current]) nếu có prev
        - cum_wspm: running sum theo station

        Với event đầu tiên: lag_1/lag_2/rolling_2 = 0 (theo yêu cầu demo).
        """
        station = str(row.get("station") or "")
        history = self._history_by_station[station]
        prev1 = history[-1] if len(history) >= 1 else None
        prev2 = history[-2] if len(history) >= 2 else None

        enriched = dict(row)

        # cum_wspm (giống Spark: sum(WSPM) over time)
        wspm = self._safe_float(row.get("WSPM"), 0.0)
        self._cum_wspm_by_station[station] += wspm
        enriched["cum_wspm"] = float(self._cum_wspm_by_station[station])

        # Tính lag/rolling cho numeric keys có trong row (tránh station/wd/time cols)
        excluded = {"No", "year", "month", "day", "hour", "station", "wd"}
        for key, value in row.items():
            if key in excluded:
                continue
            # Chỉ làm cho numeric
            cur = self._safe_float(value, 0.0)
            p1 = self._safe_float(prev1.get(key), 0.0) if prev1 else 0.0
            p2 = self._safe_float(prev2.get(key), 0.0) if prev2 else 0.0

            enriched[f"{key}_lag_1"] = float(p1) if prev1 else 0.0
            enriched[f"{key}_lag_2"] = float(p2) if prev2 else 0.0
            enriched[f"{key}_rolling_2"] = float((p1 + cur) / 2.0) if prev1 else 0.0

        # Cập nhật history (lưu raw values để tính lag cho lần sau)
        history.append(dict(row))
        return enriched

    def predict(self, frame: pd.DataFrame):
        # Normalize column name PM2.5 -> PM2_5 để đồng nhất nội bộ
        base = normalize_pm25_column(frame)

        # Enrich theo thứ tự thời gian nhận được (micro-batch có thể > 1 row)
        enriched_rows = [self._enrich_one_event(row) for row in base.to_dict(orient="records")]
        enriched_df = pd.DataFrame(enriched_rows)

        # Physical features (giống batch ETL) - chạy sau enrich để có hour/month/TEMP/DEWP sẵn.
        enriched_df = add_physical_features(enriched_df)

        infer_df = prepare_inference_frame(enriched_df, self.features, self.model_name).fillna(0)
        if self.model_name == "xgboost":
            dmatrix = xgb.DMatrix(infer_df)
            return self._model.predict(dmatrix)
        return self._model.predict(infer_df)
