"""
benchmark.py

Basic latency/throughput measurement for the /predict endpoint, run against
a live server. Not a load-testing tool (no concurrency) -- just enough to
document expected single-worker request latency and a naive requests/sec
ceiling for the README, per the M4 rubric ("basic latency/throughput
awareness").

Usage:
    python api/app.py &          # start the server first
    python api/benchmark.py
"""
import time
import statistics
import urllib.request
import json

URL = "http://127.0.0.1:5000/predict"
PAYLOAD = json.dumps({
    "pickup_datetime": "2025-03-15T18:30:00",
    "pickup_lat": 12.95, "pickup_lon": 77.60,
    "dropoff_lat": 12.98, "dropoff_lon": 77.63,
    "weather": "rain", "temperature_c": 24.0,
}).encode()

N_REQUESTS = 200


def one_request():
    req = urllib.request.Request(
        URL, data=PAYLOAD, headers={"Content-Type": "application/json"}, method="POST"
    )
    start = time.perf_counter()
    with urllib.request.urlopen(req) as resp:
        resp.read()
    return (time.perf_counter() - start) * 1000  # ms


def main():
    latencies = [one_request() for _ in range(N_REQUESTS)]
    latencies.sort()

    def pct(p):
        idx = min(int(len(latencies) * p / 100), len(latencies) - 1)
        return latencies[idx]

    total_s = sum(latencies) / 1000
    print(f"Requests: {N_REQUESTS} (sequential, single client, single Flask dev-server worker)")
    print(f"Mean latency:   {statistics.mean(latencies):.2f} ms")
    print(f"Median latency: {statistics.median(latencies):.2f} ms")
    print(f"p95 latency:    {pct(95):.2f} ms")
    print(f"p99 latency:    {pct(99):.2f} ms")
    print(f"Min / Max:      {min(latencies):.2f} / {max(latencies):.2f} ms")
    print(f"Naive throughput ceiling (sequential): {N_REQUESTS / total_s:.1f} req/s")
    print(
        "\nNote: this is Flask's single-threaded development server, deliberately used "
        "as-is for this assignment. For real throughput beyond a demo, run behind a "
        "production WSGI server (gunicorn/uvicorn workers) with multiple worker "
        "processes -- see README 'Productionizing' notes."
    )


if __name__ == "__main__":
    main()
