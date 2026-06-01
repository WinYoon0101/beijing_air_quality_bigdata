import argparse
import os
from pathlib import Path
import runpy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run realtime ingestion into Bronze")
    parser.add_argument("--mock", action="store_true", help="Generate mocked realtime payload instead of calling APIs")
    parser.add_argument("--station", help="Preferred station for mocked payload generation")
    parser.add_argument("--seed", type=int, help="Random seed for mocked payload noise")
    args = parser.parse_args()

    os.environ["INGEST_MODE"] = "api_mock" if args.mock else "api"
    if args.station:
        os.environ["MOCK_STATION"] = args.station
    if args.seed is not None:
        os.environ["MOCK_SEED"] = str(args.seed)

    runpy.run_path(str(Path(__file__).with_name("1_ingestion_minio.py")), run_name="__main__")
