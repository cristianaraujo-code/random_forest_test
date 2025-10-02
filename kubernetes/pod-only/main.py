import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import time
import logging
from download_model import MODEL_LOCAL_PATH  # import path

logging.basicConfig(level=logging.INFO)

logging.info("Loading model...")
model = joblib.load(MODEL_LOCAL_PATH)
logging.info("Model loaded successfully.")

app = FastAPI(title="RandomForest API", version="1.0")

class PredictionRequest(BaseModel):
    instances: list[list[float]]

@app.post("/v1/models/sklearn-model:predict")
async def predict(request: PredictionRequest):
    X = np.array(request.instances)

    start_time = time.perf_counter()
    preds = model.predict(X)
    end_time = time.perf_counter()
    inference_time = (end_time - start_time) * 1000  # ms

    logging.info(f"Inference completed in {inference_time:.3f} ms for {len(X)} instances.")

    return {
        "predictions": preds.tolist(),
        "inference_time_ms": round(inference_time, 3),
        "num_instances": len(X),
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)