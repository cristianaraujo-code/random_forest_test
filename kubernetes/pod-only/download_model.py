# save this as download_model.py
from minio import Minio
import os
import logging

logging.basicConfig(level=logging.INFO)

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "172.22.10.164:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin123")
BUCKET_NAME = os.getenv("BUCKET_NAME", "k8s-models")
MODEL_OBJECT_NAME = os.getenv("MODEL_OBJECT_NAME", "model.joblib")
MODEL_LOCAL_PATH = "/app/model.joblib"

client = Minio(
    MINIO_ENDPOINT,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=False,
)

logging.info("Downloading model from MinIO...")
client.fget_object(BUCKET_NAME, MODEL_OBJECT_NAME, MODEL_LOCAL_PATH)
logging.info("Model downloaded successfully to %s", MODEL_LOCAL_PATH)