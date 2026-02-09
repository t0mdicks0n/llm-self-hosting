"""Quick smoke test for vLLM serving endpoint.

Usage:
    python scripts/test_inference.py                    # defaults to localhost:8000
    python scripts/test_inference.py --host 10.0.0.1    # custom host
    python scripts/test_inference.py --port 8080        # custom port
"""

import argparse
import json
import sys
import time
import urllib.request
import urllib.error


def check_health(base_url: str) -> bool:
    """Check if the vLLM server is up and serving."""
    try:
        req = urllib.request.Request(f"{base_url}/v1/models")
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
            models = [m["id"] for m in data.get("data", [])]
            print(f"Server is up. Models available: {models}")
            return True
    except urllib.error.URLError as e:
        print(f"Server not reachable: {e}")
        return False


def run_inference(base_url: str, prompt: str, max_tokens: int = 1024) -> dict:
    """Send a chat completion request and return the response."""
    payload = json.dumps({
        "model": "Kimi-K2.5",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
    }).encode()

    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    start = time.time()
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read())
    elapsed = time.time() - start

    return {
        "response": data["choices"][0]["message"]["content"],
        "usage": data.get("usage", {}),
        "elapsed_seconds": round(elapsed, 2),
    }


def main():
    parser = argparse.ArgumentParser(description="Test vLLM inference endpoint")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"
    print(f"Testing vLLM endpoint at {base_url}\n")

    # Health check
    if not check_health(base_url):
        print("\nServer is not ready. Is vLLM still loading the model?")
        sys.exit(1)

    # Test prompts
    prompts = [
        "What is 2 + 2? Answer in one word.",
        "Write a Python function that checks if a number is prime. Keep it short.",
        "Explain the difference between tensor parallelism and pipeline parallelism in one paragraph.",
    ]

    for i, prompt in enumerate(prompts, 1):
        print(f"\n--- Test {i}/{len(prompts)} ---")
        print(f"Prompt: {prompt}")
        try:
            result = run_inference(base_url, prompt)
            response = result['response'] or "(empty - reasoning tokens may have consumed the budget)"
            print(f"Response: {response[:500]}")
            print(f"Tokens: {result['usage']}")
            print(f"Time: {result['elapsed_seconds']}s")
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)

    print("\nAll tests passed.")


if __name__ == "__main__":
    main()
