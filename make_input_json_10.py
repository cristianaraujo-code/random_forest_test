# make_input_json_10.py
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import load

from preprocessing import preprocess_dataframe, build_preprocessor


def load_data(test_path: str) -> pd.DataFrame:
    """Load test data and drop target column if present."""
    df = pd.read_csv(test_path)
    return df.drop(columns=["isFraud"], errors="ignore")


def prepare_features(df: pd.DataFrame, model_path: str, expected: int | None) -> np.ndarray:
    """Preprocess dataframe and align feature dimensions with the model input."""
    df = preprocess_dataframe(df)
    preprocessor = build_preprocessor(df)
    Xt = preprocessor.fit_transform(df)

    if expected is None:
        model_file = Path(model_path)
        if not model_file.exists():
            raise FileNotFoundError(
                f"{model_path} not found. Use --expected or provide model.joblib."
            )
        model = load(model_path)
        expected = int(model.n_features_in_)

    # Align dimensions
    if Xt.shape[1] > expected:
        Xt = Xt[:, :expected]
    elif Xt.shape[1] < expected:
        padding = np.zeros((Xt.shape[0], expected - Xt.shape[1]))
        Xt = np.hstack([Xt, padding])

    return Xt, expected


def write_json(Xt: np.ndarray, n: int, expected: int, out_path: str) -> None:
    """Write N rows from Xt into JSON payload for KServe inference."""
    if Xt.shape[0] >= n:
        rows = Xt[:n]
    else:
        reps = (n + Xt.shape[0] - 1) // Xt.shape[0]
        rows = np.vstack([Xt] * reps)[:n]

    payload = {"instances": rows.tolist()}

    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(
        f"Generated {out_path} with {len(rows)} instances "
        f"(each {expected} features)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate JSON inputs for KServe inference")
    parser.add_argument("--test", default="test.csv", help="Path to test CSV file")
    parser.add_argument("--model", default="model.joblib", help="Path to trained model")
    parser.add_argument("--expected", type=int, default=None, help="Expected input size")
    parser.add_argument("--out", default="input_numeric_10.json", help="Output JSON path")
    parser.add_argument("--n", type=int, default=10, help="Number of rows to export")
    args = parser.parse_args()

    df = load_data(args.test)
    Xt, expected = prepare_features(df, args.model, args.expected)
    write_json(Xt, args.n, expected, args.out)


if __name__ == "__main__":
    main()