import os
from pathlib import Path
import runpy


if __name__ == "__main__":
    model_choice = os.getenv("TRAIN_MODEL", "lightgbm").lower()
    script_name = "3a_train_xgboost.py" if model_choice == "xgboost" else "3b_train_lightgbm.py"
    runpy.run_path(str(Path(__file__).with_name(script_name)), run_name="__main__")
