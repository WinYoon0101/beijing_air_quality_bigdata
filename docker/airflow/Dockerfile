FROM apache/airflow:2.9.3-python3.11

USER root
# Đổi java 11 thành java 17 
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential gcc openjdk-17-jre-headless \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Cập nhật lại thư mục trỏ đến Java 17 của Linux
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH="${JAVA_HOME}/bin:${PATH}"

USER airflow
WORKDIR /opt/airflow/project

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

ENV PYTHONPATH=/opt/airflow/project