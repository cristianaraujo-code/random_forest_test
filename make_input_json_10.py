# make_input_json_10.py
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import load
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def safe_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", default="test.csv")
    parser.add_argument("--model", default="model.joblib")
    parser.add_argument("--expected", type=int, default=None)
    parser.add_argument("--out", default="input_numeric_10.json")
    parser.add_argument("--n", type=int, default=10)
    args = parser.parse_args()

    df = pd.read_csv(args.test)
    X = df.drop(columns=["isFraud"], errors="ignore")

    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", safe_ohe(), categorical_cols),
        ]
    )

    Xt = preprocessor.fit_transform(X)

    if args.expected is not None:
        expected = int(args.expected)
    else:
        model_path = Path(args.model)
        if not model_path.exists():
            raise FileNotFoundError(f"No existe {args.model}. Pasa --expected o coloca model.joblib en el directorio.")
        model = load(args.model)
        expected = int(model.n_features_in_)

    if Xt.shape[1] > expected:
        Xt = Xt[:, :expected]
    elif Xt.shape[1] < expected:
        padding = np.zeros((Xt.shape[0], expected - Xt.shape[1]))
        Xt = np.hstack([Xt, padding])

    n = args.n
    if Xt.shape[0] >= n:
        rows = Xt[:n]
    else:
        reps = (n + Xt.shape[0] - 1) // Xt.shape[0]
        rows = np.vstack([Xt] * reps)[:n]

    payload = {"instances": rows.tolist()}

    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"✅ Generado {args.out} con {len(rows)} instancias, cada una con {expected} features")


if __name__ == "__main__":
    main()
