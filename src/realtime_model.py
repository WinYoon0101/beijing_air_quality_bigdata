import json
import threading
from collections import defaultdict, deque
from pathlib import Path

import lightgbm as lgb
import pandas as pd
import xgboost as xgb

from config import MODEL_LIGHTGBM_PATH, MODEL_XGBOOST_PATH
from feature_schema import (
    enrich_realtime_row,
    load_lightgbm_booster,
    normalize_pm25_column,
    predict_lightgbm,
    prepare_inference_frame,
)

_PREDICTOR_LOCK = threading.Lock()
_PREDICTOR_SINGLETON: "RealtimePredictor | None" = None


class RealtimePredictor:
    def __init__(self):
        self.model_name = ""
        self.features: list[str] = []
        self.target_col = "Target_PM2.5_next_1h"
        self._model = None
        self._predict_lock = threading.Lock()
        self._history_by_station: dict[str, deque[dict]] = defaultdict(lambda: deque(maxlen=2))
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
        self.target_col = metadata.get("target_col", self.target_col)

        if self.model_name == "lightgbm":
            if not MODEL_LIGHTGBM_PATH.exists():
                raise FileNotFoundError(f"Thiếu model LightGBM: {MODEL_LIGHTGBM_PATH}")
            self._model = load_lightgbm_booster(MODEL_LIGHTGBM_PATH)
            self.features = list(self._model.feature_name()) or list(metadata["feature_columns"])
            self._model_mtime = MODEL_LIGHTGBM_PATH.stat().st_mtime
            return

        if self.model_name == "xgboost":
            if not MODEL_XGBOOST_PATH.exists():
                raise FileNotFoundError(f"Thiếu model XGBoost: {MODEL_XGBOOST_PATH}")
            model = xgb.Booster()
            model.load_model(str(MODEL_XGBOOST_PATH))
            self._model = model
            self.features = list(model.feature_names) or list(metadata["feature_columns"])
            return

        raise ValueError(f"Model chưa hỗ trợ realtime: {self.model_name}")

    def reload_model_if_changed(self):
        if self.model_name != "lightgbm":
            return
        current_mtime = MODEL_LIGHTGBM_PATH.stat().st_mtime
        cached_mtime = getattr(self, "_model_mtime", None)
        if cached_mtime is None or cached_mtime != current_mtime:
            with self._predict_lock:
                self._load_model()
                self._model_mtime = current_mtime

    def _enrich_batch(self, frame: pd.DataFrame) -> pd.DataFrame:
        base = normalize_pm25_column(frame)
        enriched_rows = []

        for raw_row in base.to_dict(orient="records"):
            station = str(raw_row.get("station") or "")
            history = self._history_by_station[station]
            prev1 = history[-1] if len(history) >= 1 else None
            prev2 = history[-2] if len(history) >= 2 else None

            wspm = float(raw_row.get("WSPM") or 0.0)
            self._cum_wspm_by_station[station] += wspm
            cum_wspm = self._cum_wspm_by_station[station]

            enriched, snapshot = enrich_realtime_row(raw_row, prev1, prev2, cum_wspm)
            history.append(snapshot)
            enriched_rows.append(enriched)

        return pd.DataFrame(enriched_rows)

    def predict(self, frame: pd.DataFrame):
        self.reload_model_if_changed()
        enriched_df = self._enrich_batch(frame)

        with self._predict_lock:
            if self.model_name == "lightgbm":
                return predict_lightgbm(self._model, enriched_df)
            infer_df = prepare_inference_frame(
                enriched_df,
                self.features,
                self.model_name,
            )
            numeric_df = infer_df.reindex(columns=self.features, fill_value=0).select_dtypes(
                include=["number"]
            )
            return self._model.predict(xgb.DMatrix(numeric_df))


def get_realtime_predictor() -> RealtimePredictor:
    global _PREDICTOR_SINGLETON
    if _PREDICTOR_SINGLETON is None:
        with _PREDICTOR_LOCK:
            if _PREDICTOR_SINGLETON is None:
                _PREDICTOR_SINGLETON = RealtimePredictor()
    return _PREDICTOR_SINGLETON
