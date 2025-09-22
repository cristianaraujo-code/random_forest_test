# main.py
import time
import random
import requests
import json
import numpy as np
import pandas as pd
from pathlib import Path
from joblib import load
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def safe_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(X):
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", safe_ohe(), categorical_cols),
        ]
    )


def main():
    # Load data and model
    df = pd.read_csv("test.csv").drop(columns=["isFraud"], errors="ignore")
    model = load("model.joblib")
    expected = int(model.n_features_in_)

    # Create preprocessor and transform data
    preprocessor = build_preprocessor(df)
    Xt = preprocessor.fit_transform(df)

    # adjust dimensions
    if Xt.shape[1] > expected:
        Xt = Xt[:, :expected]
    elif Xt.shape[1] < expected:
        padding = np.zeros((Xt.shape[0], expected - Xt.shape[1]))
        Xt = np.hstack([Xt, padding])

    # endpoint exposed by port-forward
    URL = "http://127.0.0.1:8081/v1/models/sklearn-model:predict"
    HEADERS = {"Content-Type": "application/json"}

    print(f"✅ Client ready. Sending requests to {URL} every 0.1s...\n")

    while True:
        # choose a random row already preprocessed
        idx = random.randint(0, Xt.shape[0] - 1)
        row = Xt[idx].tolist()

        payload = {"instances": [row]}

        try:
            response = requests.post(URL, headers=HEADERS, data=json.dumps(payload))
            print(f"[Row {idx}] Status {response.status_code}: {response.text}")
        except Exception as e:
            print(f"❌ Error in request: {e}")

        time.sleep(0.01)


if __name__ == "__main__":
    main()
