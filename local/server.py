import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import time
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)

# Load the model at startup
logging.info("Loading model...")
model = joblib.load("model.joblib")
logging.info("Model loaded successfully.")

# Define the app
app = FastAPI(title="RandomForest API", version="1.0")


# Input schema
class PredictionRequest(BaseModel):
    instances: list[list[float]]


@app.post("/v1/models/sklearn-model:predict")
async def predict(request: PredictionRequest):
    X = np.array(request.instances)

    # Measure inference time
    start_time = time.perf_counter()
    preds = model.predict(X)
    end_time = time.perf_counter()
    inference_time = (end_time - start_time) * 1000  # ms

    # Log to console
    logging.info(
        f"Inference completed in {inference_time:.3f} ms for {len(X)} instances."
    )

    # Return response with timing
    return {
        "predictions": preds.tolist(),
        "inference_time_ms": round(inference_time, 3),
        "num_instances": len(X),
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
