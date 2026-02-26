# FastAPI LLM Server

A Dockerized FastAPI server benchmarking two LLM inference backends on Apple Silicon.

## Architecture

```
Client
  └── FastAPI (Docker, port 8000)
        ├── POST /generate      → llama-cpp-python (inside container, GGUF)
        └── POST /generate_mlc  → MLC-LLM (native on host, port 8002, Metal GPU)
```

llama-cpp runs inside Docker alongside FastAPI. MLC-LLM runs **natively on the host** because Docker on Mac can't access the Metal GPU — FastAPI proxies requests to it via `host.docker.internal:8002`.

---

## Setup

**llama-cpp backend:**
```bash
docker build -t fastapi-llm-server .
docker run -p 8000:8000 fastapi-llm-server
```

**MLC backend (run natively first, then Docker):**
```bash
source venv/bin/activate
python3 -m mlc_llm serve HF://mlc-ai/TinyLlama-1.1B-Chat-v1.0-q4f16_1-MLC --device metal --port 8002

docker run -p 8000:8000 -e MLC_BASE_URL="http://host.docker.internal:8002" fastapi-llm-server
```

Docs at `http://localhost:8000/docs` — run benchmarks with `python3 bench.py`

---

## Benchmark Results (Apple M3, 18GB RAM)
3 warmup runs, 20 measured runs, `max_tokens=100`

| Metric | llama-cpp-python | MLC-LLM (Metal) |
|---|---|---|
| Avg latency | 1.062s | 0.861s |
| Median | 1.135s | 0.866s |
| p95 | 1.388s | 0.968s |

> MLC-LLM was ~19% faster than llama-cpp in this configuration.