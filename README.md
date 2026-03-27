# FastAPI LLM Server

A Dockerized FastAPI server integrating multiple LLM runtimes and VLMs on Apple Silicon.

## What I Built

- Connected **llama-cpp-python** and **MLC-LLM** backends to a FastAPI server
- Added **moondream2** (VLM) via llama-cpp (GGUF) and HuggingFace transformers
- Added a **Qwen2.5-VL-7B-Instruct** stub (hardware not available locally — requires ~16GB VRAM)
- Clients can specify the runtime per request: `llama-cpp-python`, `mlc-llm`, or `vllm`
- Measured and compared end-to-end latency across backends

## Endpoints

| Endpoint | Description |
|---|---|
| `POST /generate` | Unified endpoint — specify `runtime` and `model` |
| `POST /generate_hf` | moondream2 via HuggingFace (requires base64 image) |
| `POST /generate_mlc` | MLC-LLM (legacy alias) |
| `POST /generate_moondream` | moondream2 via llama-cpp (legacy alias) |
| `POST /generate_qwen` | Qwen2.5-VL stub |
| `GET /health` | Health check |

**Example request:**
```json
POST /generate
{
  "prompt": "What is the capital of France?",
  "runtime": "llama-cpp-python",
  "model": "tinyllama"
}
```

## Setup

```bash
docker build -t llm-server .
docker run -p 8001:8000 llm-server
```

For MLC-LLM (must run natively — Docker can't access Metal GPU):
```bash
python3 -m mlc_llm serve HF://mlc-ai/TinyLlama-1.1B-Chat-v1.0-q4f16_1-MLC --device metal --port 8002
docker run -p 8001:8000 -e MLC_BASE_URL="http://host.docker.internal:8002" llm-server
```

Swagger UI at `http://localhost:8001/docs`

## Benchmark Results (Apple M3, 18GB RAM)

20 measured runs, `max_tokens=100`

| Backend | Avg Latency | Median | p95 |
|---|---|---|---|
| llama-cpp-python | 1.062s | 1.135s | 1.388s |
| MLC-LLM (Metal) | 0.861s | 0.866s | 0.968s |

MLC-LLM was ~19% faster than llama-cpp in this configuration.
