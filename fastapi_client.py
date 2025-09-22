# fastapi_client.py
from fastapi import FastAPI
from pydantic import BaseModel
import requests
import json

app = FastAPI(title="Cliente FastAPI para KServe")

KSERVE_URL = "http://127.0.0.1:8080/v1/models/sklearn-model:predict"
HEADERS = {"Content-Type": "application/json"}


class KServePayload(BaseModel):
    instances: list


@app.post("/predict")
def predict(payload: KServePayload):
    try:
        response = requests.post(
            KSERVE_URL, headers=HEADERS, data=json.dumps(payload.dict())
        )
        return {"status_code": response.status_code, "response": response.json()}
    except Exception as e:
        return {"error": str(e)}
