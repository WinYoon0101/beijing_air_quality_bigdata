import os
from pathlib import Path
import runpy


if __name__ == "__main__":
    os.environ["INGEST_MODE"] = "api"
    runpy.run_path(str(Path(__file__).with_name("1_ingestion_minio.py")), run_name="__main__")
