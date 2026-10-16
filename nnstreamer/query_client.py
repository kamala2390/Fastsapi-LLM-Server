#!/usr/bin/env python3
"""Standalone stand-in for the 'mobile phone' side of the pipeline.

Simulates what fastapi_query_filter.py does inside a real NNStreamer
pipeline on a phone: take one captured frame, send it to the FastAPI
VLM server, print back the model's answer. Useful for testing the
server round-trip on a machine that doesn't have NNStreamer installed.

Usage:
    python3 query_client.py <image_path> ["your question"] [server_url]
"""
import sys
import time

from vlm_bridge import DEFAULT_PROMPT, DEFAULT_SERVER_URL, query_vlm_server


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <image_path> [\"prompt\"] [server_url]")
        sys.exit(1)

    image_path = sys.argv[1]
    prompt = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_PROMPT
    server_url = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_SERVER_URL

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    print(f"-> POST {server_url}")
    print(f"-> prompt: {prompt!r}")
    print(f"-> frame: {image_path} ({len(image_bytes)} bytes)")

    start = time.time()
    result = query_vlm_server(image_bytes, prompt=prompt, server_url=server_url)
    elapsed = time.time() - start

    print(f"\n<- model: {result.get('model')}")
    print(f"<- server latency_seconds: {result.get('latency_seconds')}")
    print(f"<- round-trip wall time: {elapsed:.3f}s")
    print(f"<- response: {result.get('response')}")


if __name__ == "__main__":
    main()
