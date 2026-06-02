import json
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import r2_score

from config import MODEL_METRICS_PATH
from evaluation_data import build_evaluation_payload

warnings.filterwarnings("ignore")


def _save_metrics(payload):
    metrics_list = [
        {
            "Model": row["Model"],
            "RMSE": row["RMSE"],
            "MAE": row["MAE"],
            "R2": row["R2 Score"],
        }
        for row in payload["models"]
    ]
    metrics_payload = {
        "generated_at": datetime.utcnow().isoformat(),
        "target_col": payload["target_col"],
        "sample_size": payload["sample_size"],
        "models": metrics_list,
    }
    metrics_path = Path(MODEL_METRICS_PATH)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n💾 ĐÃ LƯU METRICS: {metrics_path.absolute()}")


def _plot_from_payload(payload):
    print("\n🎨 Đang kết xuất các biểu đồ trực quan hóa...")
    sns.set_theme(style="whitegrid", context="talk")

    timeline = payload["timeline"]
    timestamps = timeline.get("timestamps", [])
    actual = np.array(timeline.get("actual", []))
    model_series = timeline.get("models", {})

    if len(actual):
        tail = min(150, len(actual))
        labels = timestamps[-tail:] if timestamps else list(range(tail))
        plt.figure(figsize=(16, 7))
        for name, series in model_series.items():
            values = np.array(series)[-tail:]
            plt.plot(values, label=f"{name} Forecast", alpha=0.8, linewidth=2)
        plt.plot(actual[-tail:], label="Actual (Ground Truth)", color="black", linewidth=2.5, linestyle="--")
        plt.title("PM2.5 Forecast vs Actual (Last 150 Hours)", fontsize=16, fontweight="bold")
        plt.ylabel("PM2.5 Concentration (µg/m³)")
        plt.xlabel("Time (Hours)")
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig("pm25_line_comparison.png", dpi=300)
        print(" 📈 Đã lưu: pm25_line_comparison.png")
        plt.close()

    for model in payload["models"]:
        name = model["Model"]
        scatter = model.get("scatter", {})
        actual_pts = np.array(scatter.get("actual", []))
        pred_pts = np.array(scatter.get("predicted", []))
        if not len(actual_pts) or not len(pred_pts):
            continue

        residuals = np.array(model.get("residuals", actual_pts - pred_pts))
        model_name_lower = name.lower()

        plt.figure(figsize=(8, 6))
        sns.kdeplot(residuals, fill=True, color="#e74c3c", alpha=0.5)
        plt.axvline(0, color="black", linestyle="--", linewidth=1.5, label="Ideal (Zero Error)")
        plt.title(f"Error Distribution - {name}", fontsize=14, fontweight="bold")
        plt.xlabel("Error (Actual - Predicted) [µg/m³]", fontsize=12)
        plt.ylabel("Density", fontsize=12)
        plt.legend()
        plt.tight_layout()
        error_file = f"pm25_error_dist_{model_name_lower}.png"
        plt.savefig(error_file, dpi=300)
        print(f" 📉 Đã lưu: {error_file}")
        plt.close()

        plt.figure(figsize=(8, 6))
        plt.scatter(actual_pts, pred_pts, alpha=0.5, color="#3498db", edgecolors="none", s=40)
        max_val = max(actual_pts.max(), pred_pts.max())
        plt.plot([0, max_val], [0, max_val], "r--", lw=2, label="y = x Line (Perfect Fit)")
        plt.title(
            f"Actual vs Predicted - {name}\n(R2 Score: {r2_score(actual_pts, pred_pts):.2f})",
            fontsize=14,
            fontweight="bold",
        )
        plt.xlabel("Actual Values", fontsize=12)
        plt.ylabel("Predicted Values", fontsize=12)
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        scatter_file = f"pm25_scatter_{model_name_lower}.png"
        plt.savefig(scatter_file, dpi=300)
        print(f" 🎯 Đã lưu: {scatter_file}")
        plt.close()


def evaluate_models():
    print("📊 [EVALUATION] Đang tải tập dữ liệu Gold từ MinIO...")
    payload = build_evaluation_payload(station=None, test_ratio=0.1, line_points=150)

    print("\n✅ KẾT QUẢ ĐÁNH GIÁ CHI TIẾT:")
    for row in payload["models"]:
        print(
            f"👉 {row['Model']}: RMSE={row['RMSE']:.4f} | MAE={row['MAE']:.4f} | R2={row['R2 Score']:.4f}"
        )

    _save_metrics(payload)
    _plot_from_payload(payload)
    print("\nHOÀN TẤT! Các file metrics và biểu đồ đã được lưu.")


if __name__ == "__main__":
    evaluate_models()
