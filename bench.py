import time
import statistics
import requests

PROMPT = "The meaning of life is"
N_WARMUP = 3
N_RUNS = 20

def p95(xs):
    xs = sorted(xs)
    k = int(0.95 * (len(xs)-1))
    return xs[k]

def bench(url: str, label: str):
    # warmup
    for _ in range(N_WARMUP):
        r = requests.post(url, json={"prompt": PROMPT}, timeout=300)
        r.raise_for_status()

    client_times = []
    server_times = []

    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        r = requests.post(url, json={"prompt": PROMPT}, timeout=300)
        r.raise_for_status()
        t1 = time.perf_counter()

        client_times.append(t1 - t0)

        data = r.json()
        if "latency_seconds" in data:
            server_times.append(float(data["latency_seconds"]))

    print(f"\n== {label} ==")
    print(f"Runs: {N_RUNS}")
    print(f"Client avg: {statistics.mean(client_times):.3f}s  median: {statistics.median(client_times):.3f}s  p95: {p95(client_times):.3f}s")
    if server_times:
        print(f"Server avg: {statistics.mean(server_times):.3f}s  median: {statistics.median(server_times):.3f}s")

def main():
    bench("http://localhost:8000/generate", "llama-cpp-python")
    bench("http://localhost:8000/generate_mlc", "mlc (metal)")

if __name__ == "__main__":
    main()