# Hướng dẫn test dashboard và realtime mock

Tài liệu này mô tả cách giả lập dữ liệu realtime theo station, chạy lại luồng hourly, và kiểm tra dashboard với tab so sánh model mới.

## 1. Chuẩn bị môi trường

```powershell
docker-compose up -d
```

Nếu bạn chạy local không qua Docker, hãy đảm bảo MinIO, Cassandra và các service cần thiết đang chạy, và `src/config.py` trỏ đúng endpoint.

## 2. Đưa dữ liệu lịch sử vào Bronze

Chạy một lần để có lịch sử trong MinIO:

```powershell
cd src
python 0_data_preprocessing.py
python 1_ingestion_minio.py
```

Ghi chú:
- `0_data_preprocessing.py` tạo file `data/airquality_data.csv` từ raw CSV.
- `1_ingestion_minio.py` ở chế độ mặc định sẽ đẩy file lịch sử đã chuẩn hóa lên Bronze.

## 3. Tạo Gold batch

```powershell
python 0_batch_etl.py
```

Sau bước này bạn phải có:
- `silver/cleaned_data.parquet`
- `gold/features.parquet`

## 4. Giả lập realtime theo station

Script này sẽ tạo payload realtime giả lập từ lịch sử của một station được chọn, rồi ghi vào Bronze realtime:

```powershell
python 1_ingest_api.py --mock --station "Aotizhongxin" --seed 42
```

Bạn có thể thay `Aotizhongxin` bằng bất kỳ station nào đang có trong Gold.

Nếu muốn xem danh sách station có sẵn, mở API sau khi chạy FastAPI hoặc kiểm tra từ endpoint:

```text
GET http://localhost:8000/api/stations
```

## 5. Tạo snapshot realtime và dự báo

Sau khi đã có payload mock, chạy tiếp:

```powershell
python 2_hourly_etl.py
python 4_realtime_inference.py --model xgboost
```

Hoặc đổi model:

```powershell
python 4_realtime_inference.py --model lightgbm
```

Kết quả sẽ được ghi vào Cassandra để dashboard đọc.

## 6. Tạo metrics summary

```powershell
python 5_evaluate_visualize.py
```

Bước này tạo `src/model_metrics.json` và file ảnh so sánh ở thư mục gốc.

## 7. Mở dashboard

Khởi động FastAPI:

```powershell
python main.py
```

Mở dashboard ở:

```text
http://localhost:8000
```

### Tab Dự báo giờ tiếp theo
- Chọn station từ dropdown.
- Bấm `Refresh` nếu muốn tải lại dữ liệu forecast.
- Nếu chưa có dữ liệu, hãy chạy lại bước mock realtime + hourly ETL + inference.

### Tab So sánh model
Tab này sẽ gọi:

```text
GET /api/model-comparison?station=<station>
```

và hiển thị cho từng model:
- Scatter Plot `Actual vs Predicted`
- Biểu đồ phân phối lỗi (residual histogram)
- Biểu đồ biến động PM2.5 theo thời gian thực tế và dự báo của 3 mô hình tại station đang chọn

## 8. Chu trình test nhanh nhất

Nếu bạn chỉ muốn test nhanh dashboard từ đầu đến cuối, chạy theo thứ tự này:

```powershell
cd src
python 0_data_preprocessing.py
python 1_ingestion_minio.py
python 0_batch_etl.py
python 1_ingest_api.py --mock --station "Aotizhongxin" --seed 42
python 2_hourly_etl.py
python 4_realtime_inference.py --model lightgbm
python 5_evaluate_visualize.py
python main.py
```

Sau đó mở dashboard, chọn station và chuyển sang tab so sánh model.

## 9. Lỗi thường gặp

- `No station found`: Gold chưa có dữ liệu station hoặc `/api/stations` chưa đọc được MinIO.
- `Không có forecast trong Cassandra`: chưa chạy `2_hourly_etl.py` hoặc `4_realtime_inference.py`.
- Tab so sánh model trống: chưa train model, hoặc file model không tồn tại tại `src/`.
- Realtime mock nhưng dashboard vẫn trống: chạy lại `2_hourly_etl.py` rồi `4_realtime_inference.py` sau khi mock ingest.