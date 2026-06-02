import json
import time

from kafka import KafkaProducer

from config import KAFKA_BOOTSTRAP_SERVERS, KAFKA_TOPIC


DEFAULT_NUM_ROWS = 1000
DEFAULT_SECONDS_PER_ROW = 0.05


def _gen_one_event(idx: int, base_ts):
    # base_ts: datetime (import local) but we avoid importing datetime here for speed
    # We'll reconstruct with pytz-free naive fields.
    year = base_ts.year
    month = base_ts.month
    day = base_ts.day
    hour = base_ts.hour

    # Nhấn mạnh tính hợp lệ schema hơn là chất lượng dữ liệu (demo).
    station = "Aotizhongxin"
    wd_values = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    wd = wd_values[idx % len(wd_values)]

    # Range giả lập hợp lý (không cần chính xác theo thực tế).
    wspm = 0.5 + (idx % 30) * 0.15
    temp = 10 + (idx % 20) * 0.8
    pres = 980 + (idx % 50) * 0.4
    dewp = -15 + (idx % 30) * 0.6
    rain = 0.0 if idx % 7 else 0.8

    # PM2.5 hơi tương quan ngẫu nhiên theo WSPM/temp/dewp để demo chart có biến thiên.
    pm25 = 40 + (idx % 50) * 0.6 + (temp - 15) * 0.25 - (dewp + 5) * 0.08
    pm10 = pm25 * 1.8 + (idx % 20) * 0.4

    # Các khí khác
    so2 = 5 + (idx % 30) * 0.2
    no2 = 20 + (idx % 60) * 0.25
    co = 0.4 + (idx % 20) * 0.03
    o3 = 30 + (idx % 70) * 0.22

    return {
        "No": int(idx),
        "year": int(year),
        "month": int(month),
        "day": int(day),
        "hour": int(hour),
        "PM2.5": float(max(pm25, 0.0)),
        "PM10": float(max(pm10, 0.0)),
        "SO2": float(max(so2, 0.0)),
        "NO2": float(max(no2, 0.0)),
        "CO": float(max(co, 0.0)),
        "O3": float(max(o3, 0.0)),
        "TEMP": float(temp),
        "PRES": float(pres),
        "DEWP": float(dewp),
        "RAIN": float(rain),
        "wd": wd,
        "WSPM": float(wspm),
        "station": station,
    }


def _to_json_bytes(payload: dict) -> bytes:
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


def run_simulator(num_rows: int = DEFAULT_NUM_ROWS, seconds_per_row: float = DEFAULT_SECONDS_PER_ROW):
    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        value_serializer=_to_json_bytes,
        linger_ms=50,
    )

    print(
        f"[SIMULATOR] Bắt đầu gửi dữ liệu lên topic '{KAFKA_TOPIC}' "
        f"tại {KAFKA_BOOTSTRAP_SERVERS} ({seconds_per_row}s/dòng). "
        f"Tổng {num_rows} dòng synthetic."
    )
    sent_count = 0

    # Sinh chuỗi thời gian theo giờ liên tiếp.
    from datetime import datetime, timedelta

    base = datetime(2026, 1, 1, 0, 0, 0)
    for idx in range(num_rows):
        ts = base + timedelta(hours=idx)
        event = _gen_one_event(idx, ts)
        producer.send(KAFKA_TOPIC, event)
        sent_count += 1
        if sent_count % 500 == 0:
            producer.flush()
            print(f"[SIMULATOR] Đã gửi {sent_count} dòng...")
        time.sleep(seconds_per_row)

    producer.flush()
    producer.close()
    print(f"[SIMULATOR] Hoàn tất. Tổng số dòng đã gửi: {sent_count}")


if __name__ == "__main__":
    # Mặc định: 1000 dòng synthetic để tránh thiếu history khi realtime/batch chỉ có 1 event.
    run_simulator()
