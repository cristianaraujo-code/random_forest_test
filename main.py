# main.py

import json
import random
import time

import numpy as np
import pandas as pd
import requests
from joblib import load
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    StandardScaler,
)

from preprocessing import preprocess_dataframe, build_preprocessor


def main() -> None:
    """Client that streams random requests to FastAPI prediction endpoint."""
    # Load data and model
    df = pd.read_csv("test.csv").drop(columns=["isFraud"], errors="ignore")
    df = preprocess_dataframe(df)

    model = load("model.joblib")
    expected = int(model.n_features_in_)

    # Preprocess
    preprocessor = build_preprocessor(df)
    Xt = preprocessor.fit_transform(df)

    # Align features with model input
    if Xt.shape[1] > expected:
        Xt = Xt[:, :expected]
    elif Xt.shape[1] < expected:
        padding = np.zeros((Xt.shape[0], expected - Xt.shape[1]))
        Xt = np.hstack([Xt, padding])

    # FastAPI endpoint
    url = "http://127.0.0.1:8000/predict"  # change if running remotely
    headers = {"Content-Type": "application/json"}

    print(f"Client ready. Sending requests every 0.1s to {url}...\n")

    while True:
        idx = random.randint(0, Xt.shape[0] - 1)
        row = Xt[idx].tolist()
        payload = {"instances": [row]}

        try:
            response = requests.post(url, headers=headers, data=json.dumps(payload))
            resp_json = response.json()
            predictions = resp_json.get("response", {}).get("predictions", [])
            print(f"[Row {idx}] -> Predictions: {predictions}")
        except Exception as e:
            print(f"Request error: {e}")

        time.sleep(0.1)


if __name__ == "__main__":
    main()