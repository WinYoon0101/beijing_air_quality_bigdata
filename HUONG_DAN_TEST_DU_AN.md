# Hướng dẫn test dự án PM2.5 Bắc Kinh

Tài liệu này mô tả cách kiểm tra toàn bộ hệ thống từ lúc khởi động hạ tầng đến lúc xem dữ liệu trên dashboard. Mục tiêu là test được từng thành phần riêng lẻ và test được cả luồng end-to-end.

## 1. Mục tiêu test

Khi test dự án, cần xác nhận các điểm sau:

- MinIO hoạt động và lưu được Bronze / Silver / Gold.
- Airflow mở được giao diện web và DAG chạy đúng.
- `5_evaluate_visualize.py` tạo `model_metrics.json` và biểu đồ PNG.
- FastAPI dashboard hiển thị so sánh MAE, RMSE, R², scatter và timeline (giống script evaluate).

---

## 2. Link truy cập nhanh

Nếu bạn chạy stack bằng `docker-compose.yml`, các service sẽ mở ở các port sau:

| Service | Link | Ghi chú |
|---|---|---|
| MinIO Console | http://localhost:9001 | Đăng nhập bằng `admin` / `password123` nếu chưa đổi |
| MinIO API | http://localhost:9000 | Endpoint S3 nội bộ cho Spark / Python |
| Airflow | http://localhost:8081 | Đăng nhập mặc định thường là `admin` / `admin` theo cấu hình init |
| Grafana | http://localhost:3000 | Tài khoản mặc định thường là `admin` / `admin` nếu chưa đổi |
| FastAPI Dashboard | http://localhost:8000 | Trang dashboard chính của dự án |
| Spark Master UI | http://localhost:8080 | Xem trạng thái Spark master |
| Cassandra | localhost:9042 | Không có giao diện web; dùng `cqlsh` hoặc tool CQL client |

---

## 3. Khởi động hạ tầng

Chạy toàn bộ stack bằng Docker Compose:

```powershell
docker-compose up -d
```

Nếu bạn muốn build lại image sau khi sửa Dockerfile hoặc code container:

```powershell
docker-compose up -d --build
```

Sau đó kiểm tra container đang chạy:

```powershell
docker ps
```

---

## 4. Kiểm tra từng service

### 4.1 MinIO

Mở:

```text
http://localhost:9001
```
(Tài khoản: admin / password123)

Nên thấy các bucket và folder như:

- `bronze/`
- `silver/`
- `gold/`

Cần xác nhận:

- Có thể đăng nhập vào MinIO Console.
- Có thể thấy bucket `air-quality-lake` hoặc bucket tương ứng trong `src/config.py`.
- File `bronze/airquality_data.csv` xuất hiện sau bước ingest lịch sử.
- File `gold/features.parquet` xuất hiện sau ETL.

### 4.2 Airflow

Mở:

```text
http://localhost:8081
```
(Tài khoản: admin / admin)

Cần xác nhận:

- Đăng nhập được vào Airflow.
- Thấy DAG `pm25_historical_training`.
- DAG không bị lỗi parse.
- Có thể trigger DAG thủ công nếu muốn test nhanh.

### 4.3 Cassandra

Cassandra không có UI web, nên kiểm tra bằng command line hoặc bằng tool CQL.

Ví dụ dùng `cqlsh` trong container:

```powershell
docker exec -it cassandra_db cqlsh
```

Sau đó kiểm tra keyspace / table (thay `<keyspace>` bằng giá trị trong `src/config.py`):

```sql
DESCRIBE KEYSPACES;
SELECT * FROM <keyspace>.pm25_forecast_hourly LIMIT 5;
```

Nếu dùng DBeaver / DataGrip / TablePlus, kết nối tới:

- Host: `localhost`
- Port: `9042`
- Keyspace: theo cấu hình trong `src/config.py`

Cần xác nhận:

- Keyspace được tạo thành công.
- Table forecast tồn tại.
- Sau khi chạy inference, có bản ghi mới với `station`, `forecast_timestamp`, `observed_timestamp`, `predicted`, `model_name`.

### 4.4 Grafana

Mở:

```text
http://localhost:3000
```

Cần xác nhận:

- Đăng nhập được vào Grafana.
- Cassandra datasource đã được cài từ plugin `hadesarchitect-cassandra-datasource`.
- Dashboard có thể đọc dữ liệu forecast.
- Biểu đồ / table hiển thị được dữ liệu từ Cassandra.

Nếu cần đăng nhập mặc định và chưa thay đổi cấu hình, thường là:

- Username: `admin`
- Password: `admin123`

### 4.5 FastAPI Dashboard

Mở:

```text
http://localhost:8000
```

Cần xác nhận:

- Dashboard HTML tải được.
- Tab dự báo giờ tiếp theo mở được.
- Tab so sánh model mở được.
- Có thể chọn station từ dropdown.
- Nút refresh hoặc reload dữ liệu hoạt động.

### 4.6 Spark Master UI

Mở:

```text
http://localhost:8082
```

Cần xác nhận:

- Spark master đang chạy.
- Worker đã connect vào master.
- Các job Spark ETL có thể chạy mà không lỗi kết nối.

---

## 5. Chu trình test đầy đủ từ đầu tới cuối

### Bước 1: Tiền xử lý dữ liệu raw

Chạy:

```powershell
cd src
python 0_data_preprocessing.py
```

Kết quả cần kiểm tra:

- File `data/airquality_data.csv` được tạo ra.
- Dữ liệu đã được chuẩn hóa cột `PM2_5`.
- Dữ liệu đã được sắp xếp theo thời gian.

### Bước 2: Đẩy dữ liệu lịch sử lên MinIO Bronze

Chạy:

```powershell
python 1_ingestion_minio.py
```

Kết quả cần kiểm tra:

- File lịch sử xuất hiện trong MinIO Console.
- Đường dẫn Bronze lịch sử được ghi thành công.

### Bước 3: Tạo Gold batch

Chạy:

```powershell
python 0_batch_etl.py
```

Kết quả cần kiểm tra:

- `silver/cleaned_data.parquet` được tạo.
- `gold/features.parquet` được tạo.
- Cột target dự báo giờ kế tiếp tồn tại.

### Bước 4: Train model

Chạy từng model theo biến môi trường hoặc file train tương ứng.

Ví dụ với LightGBM:

```powershell
set TRAIN_MODEL=lightgbm
python 3_train_model.py
```

Ví dụ với XGBoost:

```powershell
set TRAIN_MODEL=xgboost
python 3_train_model.py
```

Ví dụ với LSTM:

```powershell
set TRAIN_MODEL=lstm
python 3_train_model.py
```

Kết quả cần kiểm tra:

- File model được lưu trong `src/`.
- File metadata được tạo ra để inference dùng đúng schema.
- Không có lỗi thiếu feature hoặc lệch cột.

### Bước 5: Chạy đánh giá model

Chạy:

```powershell
python 5_evaluate_visualize.py
```

Kết quả cần kiểm tra:

- File `src/model_metrics.json` được tạo.
- Có MAE, RMSE, R2 cho từng model.
- Có file ảnh so sánh dự báo và thực tế nếu script xuất plot.

### Bước 6: Mở API dashboard

Chạy:

```powershell
python main.py
```

Sau đó mở:

```text
http://localhost:8000
```

Kết quả cần kiểm tra:

- Dashboard tải thành công với biểu đồ MAE, RMSE, R².
- Có đường Actual vs Predicted và scatter / phân phối lỗi từng model.
- API `GET /api/stations` và `GET /api/model-comparison` trả dữ liệu hợp lệ.

---

## 6. Cách test nhanh nhất cho toàn bộ hệ thống

Nếu bạn muốn test từ đầu tới cuối trong một luồng ngắn, chạy theo thứ tự này:

```powershell
cd src
python 0_data_preprocessing.py
python 1_ingestion_minio.py
python 0_batch_etl.py
set TRAIN_MODEL=lightgbm
python 3_train_model.py
python 5_evaluate_visualize.py
python main.py
```

Sau đó mở các link sau để kiểm tra:

- MinIO: http://localhost:9001
- Airflow: http://localhost:8081
- Grafana: http://localhost:3000
- Dashboard: http://localhost:8000

---

## 7. Các URL và endpoint nên kiểm tra trong trình duyệt

### MinIO

- `http://localhost:9001`

### Airflow

- `http://localhost:8081`

### Grafana

- `http://localhost:3000`

### Dashboard

- `http://localhost:8000`

### FastAPI endpoints

- `http://localhost:8000/health`
- `http://localhost:8000/api/stations`
- `http://localhost:8000/api/model-comparison`
- `http://localhost:8000/api/model-comparison?station=Aotizhongxin`
- `http://localhost:8000/api/metrics-file`

---

## 8. Dấu hiệu test thành công

Một lần test được xem là thành công khi:

- MinIO có đủ file Bronze / Gold.
- Airflow mở được và DAG `pm25_historical_training` chạy được.
- `model_metrics.json` và biểu đồ PNG được tạo sau evaluate.
- Dashboard FastAPI hiển thị so sánh model đầy đủ.

---

## 9. Lỗi thường gặp và cách xử lý nhanh

### Không vào được MinIO

- Kiểm tra container `minio_datalake` có chạy không.
- Kiểm tra port `9001` có bị chiếm không.

### Airflow mở nhưng không thấy DAG

- Kiểm tra `airflow/dags/pm25_orchestration_dag.py`.
- Kiểm tra container `airflow-webserver` và `airflow-scheduler`.

### Dashboard trống dữ liệu

- Kiểm tra MinIO có `gold/features.parquet` và các file model trong `src/`.
- Chạy `python 5_evaluate_visualize.py` trước khi mở dashboard.
- Kiểm tra API `GET /api/model-comparison` trong trình duyệt.

### Grafana không hiển thị gì

- Kiểm tra datasource Cassandra đã được cài.
- Kiểm tra query panel có trỏ đúng keyspace/table.

---

## 10. Ghi chú thực tế khi demo

Nếu bạn cần demo nhanh trước lớp hoặc trước team, chỉ cần chạy theo thứ tự:

1. `docker-compose up -d`
2. `python 0_data_preprocessing.py`
3. `python 1_ingestion_minio.py`
4. `python 0_batch_etl.py`
5. `set TRAIN_MODEL=lightgbm` rồi `python 3_train_model.py`
6. `python 5_evaluate_visualize.py`
7. `python main.py`

Sau đó mở các link:

- MinIO: http://localhost:9001
- Airflow: http://localhost:8081
- Dashboard: http://localhost:8000

Nếu cần, xem thêm `README.md` và `THUYET_TRINH_DU_AN.md`.
