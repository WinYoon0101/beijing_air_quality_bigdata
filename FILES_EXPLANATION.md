# Giải thích các file trong dự án

Dưới đây là tóm tắt vai trò của từng file/folder chính trong workspace.

- `docker-compose.yml`: cấu hình Docker Compose khởi tạo MinIO, Spark master/worker, Cassandra, Grafana và Airflow.
- `requirements.txt`: danh sách thư viện Python cần cài (pandas, pyspark, lightgbm, torch,...).
- `run_pipeline.bat`: batch script Windows cung cấp menu tương tác để khởi động Docker, chạy pipeline, huấn luyện và inference.
- `src/dashboard.html`: giao diện web dashboard hiện đại với tab forecast và tab so sánh model.

Folder `data/`:
- `data/raw/`: chứa các file CSV thô (ví dụ PRSA_Data_*.csv).
- `data/airquality_data.csv`: file đầu ra sau bước tiền xử lý nằm ở level local (được tạo bởi `0_data_preprocessing.py`).

Folder `src/`:
- `0_data_preprocessing.py`: gộp các CSV thô, xử lý ngoại lai, mã hóa biến phân loại, nội suy giá trị thiếu và lưu ra `data/airquality_data.csv`.
- `1_ingestion_minio.py`: upload file đã xử lý lên MinIO vào đường dẫn `bronze/airquality_data.csv`.
- `2_spark_etl.py`: dùng PySpark để đọc Bronze layer từ MinIO (`s3a://...`), tính lag/rolling feature, tính biến dẫn xuất và lưu Parquet vào Gold layer.
- `0_batch_etl.py`: wrapper cho batch historical flow theo tên flow mới.
- `1_ingest_api.py`: wrapper ingest realtime API vào Bronze.
- `2_hourly_etl.py`: tạo snapshot feature mới nhất cho realtime forecast.
- `3_train_model.py`: wrapper chọn XGBoost hoặc LightGBM để huấn luyện.
- `4_realtime_inference.py`: wrapper cho realtime inference T+1.
- `3a_train_xgboost.py`: huấn luyện mô hình XGBoost từ Gold layer, lưu model `model_xgboost_pm25.json`.
- `3b_train_lightgbm.py`: huấn luyện LightGBM, lưu model `model_lightgbm_pm25.txt`.
- `3c_train_lstm.py`: huấn luyện mô hình LSTM bằng PyTorch, lưu checkpoint `model_lstm_pm25.pth` cùng metadata feature.
- `4_batch_inference.py`: đọc Gold, load model LightGBM, dự báo trên các bản ghi gần nhất và ghi kết quả vào Cassandra.
- `main.py`: FastAPI backend cho dashboard, đọc forecast từ Cassandra.
- `main.py`: FastAPI backend cho dashboard, đọc forecast từ Cassandra và serve web app.
- `airflow/dags/pm25_orchestration_dag.py`: định nghĩa DAG batch historical và DAG realtime hourly.
- `5_evaluate_visualize.py`: script để đánh giá và vẽ biểu đồ (sử dụng matplotlib/seaborn). (Xem file để biết chi tiết thực thi).
- `config.py`: cấu hình endpoints, credentials và đường dẫn dùng chung cho các script.

File model:
- `model_lightgbm_pm25.txt`, `model_xgboost_pm25.json`, `model_lstm_pm25.pth`: các file model đã được lưu sẵn (nếu có) trong repo.

Ghi chú quan trọng:
- Một số file (`config.py`, `docker-compose.yml`) hiện chứa credentials mẫu. Nên thay bằng biến môi trường hoặc file `.env` trước khi đưa lên repo công khai.
- PySpark trên Windows yêu cầu cài đặt JDK 11 và cấu hình `HADOOP_HOME` (đã có cấu hình tạm trong `2_spark_etl.py`).
