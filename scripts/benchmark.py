"""Basic latency and throughput benchmark for vLLM endpoint.

Measures:
  - Time to first token (TTFT)
  - End-to-end latency
  - Tokens per second (output)
  - Throughput under concurrent load

Usage:
    python scripts/benchmark.py                             # quick benchmark
    python scripts/benchmark.py --concurrent 4 --rounds 5   # heavier load test
    python scripts/benchmark.py --host 10.0.0.1             # custom host
"""

import argparse
import json
import statistics
import sys
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed


def single_request(base_url: str, prompt: str, max_tokens: int = 128) -> dict:
    """Send one request and measure timing."""
    payload = json.dumps({
        "model": "Kimi-K2.5",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
    }).encode()

    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    start = time.time()
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = json.loads(resp.read())
    elapsed = time.time() - start

    usage = data.get("usage", {})
    output_tokens = usage.get("completion_tokens", 0)

    return {
        "elapsed_s": round(elapsed, 3),
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "output_tokens": output_tokens,
        "tokens_per_second": round(output_tokens / elapsed, 1) if elapsed > 0 and output_tokens else 0,
    }


def run_sequential_benchmark(base_url: str, rounds: int) -> list[dict]:
    """Run sequential requests to measure single-request latency."""
    prompt = "Explain what a neural network is in exactly 3 sentences."
    results = []

    for i in range(rounds):
        print(f"  Sequential request {i + 1}/{rounds}...", end=" ", flush=True)
        try:
            r = single_request(base_url, prompt)
            print(f"{r['elapsed_s']}s ({r['tokens_per_second']} tok/s)")
            results.append(r)
        except Exception as e:
            print(f"FAILED: {e}")

    return results


def run_concurrent_benchmark(base_url: str, concurrent: int, rounds: int) -> list[dict]:
    """Run concurrent requests to measure throughput under load."""
    prompt = "Write a haiku about GPUs."
    results = []
    total_requests = concurrent * rounds

    print(f"  Sending {total_requests} requests ({concurrent} concurrent x {rounds} rounds)...")

    for round_num in range(rounds):
        batch_results = []
        with ThreadPoolExecutor(max_workers=concurrent) as pool:
            futures = [pool.submit(single_request, base_url, prompt, 64) for _ in range(concurrent)]
            for f in as_completed(futures):
                try:
                    batch_results.append(f.result())
                except Exception as e:
                    print(f"    Request failed: {e}")

        results.extend(batch_results)
        avg_tps = statistics.mean(r["tokens_per_second"] for r in batch_results) if batch_results else 0
        print(f"    Round {round_num + 1}/{rounds}: avg {avg_tps:.1f} tok/s across {len(batch_results)} requests")

    return results


def print_stats(label: str, results: list[dict]):
    """Print summary statistics."""
    if not results:
        print(f"\n{label}: No results")
        return

    latencies = [r["elapsed_s"] for r in results]
    tps_values = [r["tokens_per_second"] for r in results if r["tokens_per_second"] > 0]

    print(f"\n{label}:")
    print(f"  Requests:     {len(results)}")
    print(f"  Latency (s):  min={min(latencies):.2f}  avg={statistics.mean(latencies):.2f}  "
          f"max={max(latencies):.2f}  p50={statistics.median(latencies):.2f}")
    if tps_values:
        print(f"  Tok/s:        min={min(tps_values):.1f}  avg={statistics.mean(tps_values):.1f}  "
              f"max={max(tps_values):.1f}")
    total_output = sum(r["output_tokens"] for r in results)
    total_time = sum(r["elapsed_s"] for r in results)
    print(f"  Total output: {total_output} tokens in {total_time:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="Benchmark vLLM endpoint")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--rounds", type=int, default=3, help="Number of rounds per benchmark")
    parser.add_argument("--concurrent", type=int, default=2, help="Concurrent requests for throughput test")
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"
    print(f"Benchmarking vLLM at {base_url}")
    print(f"Rounds: {args.rounds}, Concurrency: {args.concurrent}\n")

    # Check server health
    try:
        req = urllib.request.Request(f"{base_url}/v1/models")
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
            model = data["data"][0]["id"] if data.get("data") else "unknown"
            print(f"Model: {model}\n")
    except Exception as e:
        print(f"Server not reachable at {base_url}: {e}")
        sys.exit(1)

    # Sequential benchmark
    print("=== Sequential Latency ===")
    seq_results = run_sequential_benchmark(base_url, args.rounds)
    print_stats("Sequential Results", seq_results)

    # Concurrent benchmark
    print(f"\n=== Concurrent Throughput (x{args.concurrent}) ===")
    conc_results = run_concurrent_benchmark(base_url, args.concurrent, args.rounds)
    print_stats("Concurrent Results", conc_results)


if __name__ == "__main__":
    main()
