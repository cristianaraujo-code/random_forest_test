# make_input_json_10.py
import argparse
import json
from pathlib import Path
<<<<<<< HEAD

=======
>>>>>>> f190003 (The model is running over Kserve, the inference was test and pass)
import numpy as np
import pandas as pd
from joblib import load
from sklearn.compose import ColumnTransformer
<<<<<<< HEAD
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler


def safe_ohe() -> OneHotEncoder:
    """Return OneHotEncoder with backward compatibility across sklearn versions."""
=======
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def safe_ohe():
>>>>>>> f190003 (The model is running over Kserve, the inference was test and pass)
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


<<<<<<< HEAD
def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply preprocessing inspired by the Hyphatia paper (IEEE-CIS dataset).
    - Handle missing values
    - Filter frequent categories
    - Compute transaction aggregations
    - Add temporal behavior features
    """
    # Handle missing numeric values
    for col in df.select_dtypes(include=[np.number]).columns:
        if "mean_diff_transaction_dt" in col:
            df[col] = df[col].fillna(0)
        else:
            df[col] = df[col].fillna(-1)

    # Handle missing categorical values
    for col in df.select_dtypes(include=["object", "category"]).columns:
        df[col] = df[col].fillna("unknown")

    # Filter frequent values in id_31 (browser/platform)
    if "id_31" in df.columns:
        common_vals = {"chrome", "safari", "edge", "firefox", "samsung"}
        df = df[df["id_31"].isin(common_vals)]

    # Transaction amount aggregations
    if "TransactionAmt" in df.columns and "card1" in df.columns:
        df["mean_transaction_amt"] = df.groupby("card1")["TransactionAmt"].transform(
            "mean"
        )
        df["min_transaction_amt"] = df.groupby("card1")["TransactionAmt"].transform(
            "min"
        )
        df["max_transaction_amt"] = df.groupby("card1")["TransactionAmt"].transform(
            "max"
        )

    # Temporal behavior: difference between transactions
    if "TransactionDT" in df.columns and "card1" in df.columns:
        df = df.sort_values(by=["card1", "TransactionDT"])
        df["diff_transaction_dt"] = (
            df.groupby("card1")["TransactionDT"].diff().fillna(0)
        )
        df["mean_diff_transaction_dt"] = df.groupby("card1")[
            "diff_transaction_dt"
        ].transform("mean")

    return df


def main() -> None:
    """Generate a JSON file with N preprocessed rows for inference."""
=======
def main():
>>>>>>> f190003 (The model is running over Kserve, the inference was test and pass)
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", default="test.csv")
    parser.add_argument("--model", default="model.joblib")
    parser.add_argument("--expected", type=int, default=None)
    parser.add_argument("--out", default="input_numeric_10.json")
    parser.add_argument("--n", type=int, default=10)
    args = parser.parse_args()

    df = pd.read_csv(args.test)
    X = df.drop(columns=["isFraud"], errors="ignore")

<<<<<<< HEAD
    # Apply Hyphatia-style preprocessing
    X = preprocess_dataframe(X)

    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
=======
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
>>>>>>> f190003 (The model is running over Kserve, the inference was test and pass)
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
<<<<<<< HEAD
            ("num_std", StandardScaler(), numeric_cols),
            ("num_minmax", MinMaxScaler(), numeric_cols),
=======
            ("num", StandardScaler(), numeric_cols),
>>>>>>> f190003 (The model is running over Kserve, the inference was test and pass)
            ("cat", safe_ohe(), categorical_cols),
        ]
    )

    Xt = preprocessor.fit_transform(X)

    if args.expected is not None:
        expected = int(args.expected)
    else:
        model_path = Path(args.model)
        if not model_path.exists():
<<<<<<< HEAD
            raise FileNotFoundError(
                f"{args.model} not found. Use --expected or provide model.joblib."
            )
        model = load(args.model)
        expected = int(model.n_features_in_)

    # Adjust dimensions to match model input
=======
            raise FileNotFoundError(f"No existe {args.model}. Pasa --expected o coloca model.joblib en el directorio.")
        model = load(args.model)
        expected = int(model.n_features_in_)

>>>>>>> f190003 (The model is running over Kserve, the inference was test and pass)
    if Xt.shape[1] > expected:
        Xt = Xt[:, :expected]
    elif Xt.shape[1] < expected:
        padding = np.zeros((Xt.shape[0], expected - Xt.shape[1]))
        Xt = np.hstack([Xt, padding])

<<<<<<< HEAD
    # Select N rows
=======
>>>>>>> f190003 (The model is running over Kserve, the inference was test and pass)
    n = args.n
    if Xt.shape[0] >= n:
        rows = Xt[:n]
    else:
        reps = (n + Xt.shape[0] - 1) // Xt.shape[0]
        rows = np.vstack([Xt] * reps)[:n]

    payload = {"instances": rows.tolist()}

    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)

<<<<<<< HEAD
    print(
        f"✅ Generated {args.out} with {len(rows)} instances, "
        f"each with {expected} features"
    )
=======
    print(f"✅ Generado {args.out} con {len(rows)} instancias, cada una con {expected} features")
>>>>>>> f190003 (The model is running over Kserve, the inference was test and pass)


if __name__ == "__main__":
    main()
