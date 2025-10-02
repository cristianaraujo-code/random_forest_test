# plot_benchmark.py
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_PATH = "benchmark_results.csv"

def plot_results():
    df = pd.read_csv(RESULTS_PATH)

    # Plot Latency
    plt.figure(figsize=(10, 6))
    plt.plot(df["requests"], df["latency_avg_ms"], marker="o", label="Avg Latency (ms)")
    plt.fill_between(df["requests"], df["latency_min_ms"], df["latency_max_ms"], alpha=0.2, label="Min-Max Latency")
    plt.xscale("log")
    plt.xlabel("Number of Requests (log scale)")
    plt.ylabel("Latency (ms)")
    plt.title("Latency vs Requests")
    plt.legend()
    plt.grid(True)
    plt.savefig("latency_vs_requests.png")
    plt.show()

    # Plot Throughput
    plt.figure(figsize=(10, 6))
    plt.plot(df["requests"], df["throughput_rps"], marker="s", color="orange")
    plt.xscale("log")
    plt.xlabel("Number of Requests (log scale)")
    plt.ylabel("Throughput (req/sec)")
    plt.title("Throughput vs Requests")
    plt.grid(True)
    plt.savefig("throughput_vs_requests.png")
    plt.show()

    # Plot Model Inference Time (if available)
    if "model_time_avg_ms" in df and df["model_time_avg_ms"].notna().any():
        plt.figure(figsize=(10, 6))
        plt.plot(df["requests"], df["model_time_avg_ms"], marker="^", color="green", label="Model Avg (ms)")
        plt.fill_between(df["requests"], df["model_time_min_ms"], df["model_time_max_ms"], alpha=0.2, color="green", label="Min-Max")
        plt.xscale("log")
        plt.xlabel("Number of Requests (log scale)")
        plt.ylabel("Model Inference Time (ms)")
        plt.title("Model Inference Time vs Requests")
        plt.legend()
        plt.grid(True)
        plt.savefig("model_time_vs_requests.png")
        plt.show()


if __name__ == "__main__":
    plot_results()