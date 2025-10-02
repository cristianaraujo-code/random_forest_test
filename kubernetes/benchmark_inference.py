import random
import time

import numpy as np
import pandas as pd
import requests

from preprocessing import build_preprocessor, preprocess_dataframe

URL = "http://172.22.10.161:30080/v1/models/sklearn-model:predict"
TESTS = [1, 10, 100, 500, 1000, 3000, 5000, 10000, 20000, 30000, 50000, 70000, 100000]
CSV_PATH = "../test.csv"
RESULTS_PATH = "benchmark_results.csv"


def load_and_preprocess(csv_path: str, expected_features: int = 72) -> np.ndarray:
    """Load test.csv and preprocess to get features aligned with the model input."""

    df = pd.read_csv(csv_path).drop(columns=["isFraud"], errors="ignore")
    df = preprocess_dataframe(df)
    preprocessor = build_preprocessor(df)
    Xt = preprocessor.fit_transform(df)
    if Xt.shape[1] > expected_features:
        Xt = Xt[:, :expected_features]
    elif Xt.shape[1] < expected_features:
        padding = np.zeros((Xt.shape[0], expected_features - Xt.shape[1]))
        Xt = np.hstack([Xt, padding])
    return Xt


def run_benchmark(Xt: np.ndarray) -> None:
    results = []
    for n in TESTS:
        print(f"\n--- Benchmark: {n} requests ---")
        latencies = []
        model_times = []
        start_batch = time.perf_counter()
        for i in range(n):
            row = Xt[random.randint(0, Xt.shape[0] - 1)].tolist()
            payload = {"instances": [row]}
            t0 = time.perf_counter()
            r = requests.post(URL, json=payload)
            t1 = time.perf_counter()
            latency_ms = (t1 - t0) * 1000
            if r.status_code == 200:
                latencies.append(latency_ms)
                try:
                    resp = r.json()
                    if "inference_time_ms" in resp:
                        model_times.append(resp["inference_time_ms"])
                except Exception:
                    pass
            else:
                print(f"Request {i} failed: {r.status_code}")
        end_batch = time.perf_counter()
        total_time = end_batch - start_batch
        throughput = n / total_time if total_time > 0 else 0
        result = {
            "requests": n,
            "total_time_s": total_time,
            "throughput_rps": throughput,
            "latency_avg_ms": np.mean(latencies) if latencies else None,
            "latency_min_ms": np.min(latencies) if latencies else None,
            "latency_max_ms": np.max(latencies) if latencies else None,
            "model_time_avg_ms": np.mean(model_times) if model_times else None,
            "model_time_min_ms": np.min(model_times) if model_times else None,
            "model_time_max_ms": np.max(model_times) if model_times else None,
        }
        results.append(result)
        print(f"Total time: {total_time:.2f} s")
        print(f"Throughput: {throughput:.2f} req/sec")
        if latencies:
            print(
                f"Latency -> Avg: {result['latency_avg_ms']:.2f} ms | "
                f"Min: {result['latency_min_ms']:.2f} ms | "
                f"Max: {result['latency_max_ms']:.2f} ms",
            )
        if model_times:
            print(
                f"Inference time (model) -> Avg: {result['model_time_avg_ms']:.2f} ms "
                f"| Min: {result['model_time_min_ms']:.2f} ms "
                f"| Max: {result['model_time_max_ms']:.2f} ms",
            )
    df_results = pd.DataFrame(results)
    df_results.to_csv(RESULTS_PATH, index=False)
    print(f"\nResults saved to {RESULTS_PATH}")


def main() -> None:
    print("Loading and preprocessing test data...")
    Xt = load_and_preprocess(CSV_PATH, expected_features=72)
    print(f"Processed dataset shape: {Xt.shape}")
    run_benchmark(Xt)


if __name__ == "__main__":
    main()
