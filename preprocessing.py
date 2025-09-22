# preprocessing.py
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler


def safe_ohe() -> OneHotEncoder:
    """Return OneHotEncoder with backward compatibility across sklearn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


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


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build column transformer with scaling and one-hot encoding."""
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    return ColumnTransformer(
        transformers=[
            ("num_std", StandardScaler(), numeric_cols),
            ("num_minmax", MinMaxScaler(), numeric_cols),
            ("cat", safe_ohe(), categorical_cols),
        ]
    )