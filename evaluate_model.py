# evaluate_model.py
import argparse

import numpy as np
import pandas as pd
from joblib import load
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from preprocessing import preprocess_dataframe, build_preprocessor




def main() -> None:
    """Evaluate the trained fraud detection model on test data."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", default="test.csv")
    parser.add_argument("--model", default="model.joblib")
    args = parser.parse_args()

    # Load dataset
    df = pd.read_csv(args.test)

    if "isFraud" not in df.columns:
        raise ValueError("The test dataset must contain the 'isFraud' column.")

    y_true = df["isFraud"].astype(int).values
    X = df.drop(columns=["isFraud"], errors="ignore")

    # Preprocess features
    X = preprocess_dataframe(X)
    preprocessor = build_preprocessor(X)
    Xt = preprocessor.fit_transform(X)

    # Load trained model
    model = load(args.model)
    expected = int(model.n_features_in_)

    # Align features with model input
    if Xt.shape[1] > expected:
        Xt = Xt[:, :expected]
    elif Xt.shape[1] < expected:
        padding = np.zeros((Xt.shape[0], expected - Xt.shape[1]))
        Xt = np.hstack([Xt, padding])

    # Predictions
    y_pred = model.predict(Xt)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(Xt)[:, 1]
    else:
        y_proba = y_pred  # fallback if no probabilities

    # Metrics
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUROC": roc_auc_score(y_true, y_proba),
        "Macro Precision": precision_score(y_true, y_pred, average="macro"),
        "Macro Recall": recall_score(y_true, y_pred, average="macro"),
        "Macro F1": f1_score(y_true, y_pred, average="macro"),
    }

    print("Model Evaluation Results:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    print("\nDetailed classification report:")
    print(classification_report(y_true, y_pred, digits=4))


if __name__ == "__main__":
    main()