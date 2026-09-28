"""Shared HTTP bridge to the FastAPI VLM server (/generate_hf).

Used by both:
  - query_client.py          (standalone CLI, runs anywhere with just `requests`)
  - fastapi_query_filter.py  (the NNStreamer custom tensor_filter subplugin)

so the exact code path that gets exercised inside a real NNStreamer pipeline
is the same one that can be tested standalone without NNStreamer installed.
"""
import base64

import requests

DEFAULT_SERVER_URL = "http://localhost:8000/generate_hf"
DEFAULT_PROMPT = "Describe what you see in this image in one sentence."


def query_vlm_server(image_bytes: bytes, prompt: str = DEFAULT_PROMPT,
                      server_url: str = DEFAULT_SERVER_URL, timeout: float = 60.0) -> dict:
    """Send one JPEG/PNG frame to the FastAPI moondream2 endpoint and return its JSON reply.

    Mirrors exactly what an NNStreamer pipeline's appsink/tensor_filter callback
    would do with a captured camera frame: encode -> base64 -> POST -> parse text back.
    """
    encoded = base64.b64encode(image_bytes).decode("ascii")
    resp = requests.post(
        server_url,
        json={
            "prompt": prompt,
            "model": "moondream2",
            "image": encoded,
            "max_tokens": 100,
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()
