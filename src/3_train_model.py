import os
from pathlib import Path
import runpy


MODEL_SCRIPT_MAP = {
    "xgboost": "3a_train_xgboost.py",
    "lightgbm": "3b_train_lightgbm.py",
    "lstm": "3c_train_lstm.py",
}


def resolve_script(model_choice):
    normalized = (model_choice or "lightgbm").strip().lower()
    if normalized not in MODEL_SCRIPT_MAP:
        valid = ", ".join(sorted(MODEL_SCRIPT_MAP.keys()))
        raise ValueError(f"TRAIN_MODEL không hợp lệ: {model_choice}. Hỗ trợ: {valid}")
    return MODEL_SCRIPT_MAP[normalized]


if __name__ == "__main__":
    model_choice = os.getenv("TRAIN_MODEL", "lightgbm")
    script_name = resolve_script(model_choice)
    runpy.run_path(str(Path(__file__).with_name(script_name)), run_name="__main__")
