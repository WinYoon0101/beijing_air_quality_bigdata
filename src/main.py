from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse

from evaluation_data import build_evaluation_payload, load_gold_frame, load_metrics_file

app = FastAPI(title="PM2.5 Model Evaluation Dashboard", version="2.0.0")
DASHBOARD_HTML_PATH = Path(__file__).with_name("dashboard.html")


def _load_station_options():
    try:
        df = load_gold_frame()
        if "station" in df.columns:
            return sorted({str(value) for value in df["station"].dropna().astype(str).tolist()})
    except Exception:
        pass
    return []


@app.get("/", response_class=HTMLResponse)
def dashboard():
    if not DASHBOARD_HTML_PATH.exists():
        raise HTTPException(status_code=500, detail="Dashboard UI not found")
    return DASHBOARD_HTML_PATH.read_text(encoding="utf-8")


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard_alias():
    return dashboard()


@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


@app.get("/api/stations")
def stations():
    station_list = _load_station_options()
    return {"stations": station_list}


@app.get("/api/metrics-file")
def metrics_file():
    """Metrics đã lưu từ `5_evaluate_visualize.py` (toàn bộ test set)."""
    return load_metrics_file()


@app.get("/api/model-comparison")
def model_comparison(
    station: str = Query(default=""),
    test_ratio: float = Query(default=0.1, ge=0.05, le=0.4),
    line_points: int = Query(default=150, ge=30, le=500),
):
    try:
        return build_evaluation_payload(
            station=station or None,
            test_ratio=test_ratio,
            line_points=line_points,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        fallback = load_metrics_file()
        if fallback.get("models"):
            return {
                "generated_at": fallback.get("generated_at"),
                "station": "Từ file model_metrics.json",
                "target_col": fallback.get("target_col"),
                "sample_size": fallback.get("sample_size"),
                "models": [
                    {
                        "Model": row["Model"],
                        "MAE": row["MAE"],
                        "RMSE": row["RMSE"],
                        "R2 Score": row.get("R2", row.get("R2 Score")),
                        "scatter": {"actual": [], "predicted": []},
                        "residuals": [],
                    }
                    for row in fallback["models"]
                ],
                "timeline": {"timestamps": [], "actual": [], "models": {}},
                "metrics_only": True,
            }
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/forecasts/latest")
@app.get("/api/latest-forecast")
@app.get("/forecasts/history")
@app.get("/api/forecast-history")
def forecast_removed():
    raise HTTPException(
        status_code=410,
        detail="Endpoint dự báo đã gỡ. Chạy `python 5_evaluate_visualize.py` rồi mở dashboard so sánh model.",
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
