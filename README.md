# FastAPI LLM Server

A Dockerized FastAPI server supporting multiple LLM inference backends and VLMs on Apple Silicon.

## Architecture

```
Client
  └── FastAPI (Docker, port 8000)
        ├── POST /generate          → routes by runtime + model
        ├── POST /generate_hf       → HuggingFace transformers VLMs
        ├── POST /generate_mlc      → MLC-LLM (legacy alias)
        ├── POST /generate_moondream→ moondream2 via llama-cpp (legacy alias)
        └── POST /generate_qwen     → Qwen2.5-VL stub (legacy alias)
```

**Runtime routing:**
- `llama-cpp-python` — runs inside Docker (CPU/GGUF)
- `mlc-llm` — runs natively on the host (Metal GPU), proxied via `host.docker.internal:8002`
- `vllm` — runs natively on the host, proxied via `host.docker.internal:8003`

---

## Models Supported

| Model | ID | Runtimes | Notes |
|---|---|---|---|
| TinyLlama-1.1B-Chat | `tinyllama` | llama-cpp-python, mlc-llm | GGUF baked into image |
| moondream2 (text) | `moondream2` | llama-cpp-python | GGUF baked into image |
| moondream2 (VLM) | `moondream2` | huggingface (`/generate_hf`) | Downloads from HF at runtime; requires image |
| Qwen2.5-VL-7B-Instruct | `qwen2.5-vl` | — | Stub only — requires ~16 GB VRAM |

---

## Setup

### 1. llama-cpp-python (Docker)

```bash
docker build -t llm-server .
docker run -p 8001:8000 llm-server
```

### 2. MLC-LLM (native on host, then Docker)

MLC-LLM must run natively because Docker on Mac cannot access the Metal GPU.

```bash
source venv/bin/activate
python3 -m mlc_llm serve HF://mlc-ai/TinyLlama-1.1B-Chat-v1.0-q4f16_1-MLC \
    --device metal --port 8002

docker run -p 8001:8000 \
    -e MLC_BASE_URL="http://host.docker.internal:8002" \
    llm-server
```

### 3. vLLM (native on host)

```bash
pip install vllm
python -m vllm.entrypoints.openai.api_server \
    --model <hf-model-id> --port 8003

docker run -p 8001:8000 \
    -e VLLM_BASE_URL="http://host.docker.internal:8003" \
    llm-server
```

Swagger UI at `http://localhost:8001/docs`

---

## API

### Unified endpoint — `POST /generate`

Clients specify both `runtime` and `model`:

```json
{
  "prompt": "What is the capital of France?",
  "runtime": "llama-cpp-python",
  "model": "tinyllama",
  "max_tokens": 100
}
```

| Field | Options | Default |
|---|---|---|
| `runtime` | `llama-cpp-python`, `mlc-llm`, `vllm` | `llama-cpp-python` |
| `model` | `tinyllama`, `moondream2`, `qwen2.5-vl` | `tinyllama` |
| `max_tokens` | integer | `100` |

### HuggingFace VLM endpoint — `POST /generate_hf`

Runs VLMs directly via HuggingFace transformers. Requires a base64-encoded image for moondream2.

```json
{
  "prompt": "What do you see in this image?",
  "model": "moondream2",
  "image": "<base64-encoded image>"
}
```

To encode an image in Python:
```python
import base64, requests

with open("photo.jpg", "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode()

r = requests.post("http://localhost:8001/generate_hf", json={
    "prompt": "What do you see in this image?",
    "model": "moondream2",
    "image": img_b64
})
print(r.json())
```

### Legacy endpoints (backwards compatible)

| Endpoint | Equivalent to |
|---|---|
| `POST /generate_mlc` | `/generate` with `runtime=mlc-llm` |
| `POST /generate_moondream` | `/generate` with `runtime=llama-cpp-python`, `model=moondream2` |
| `POST /generate_qwen` | Returns unavailable stub |

---

## Benchmark Results (Apple M3, 18GB RAM)

3 warmup runs, 20 measured runs, `max_tokens=100`

| Metric | llama-cpp-python | MLC-LLM (Metal) |
|---|---|---|
| Avg latency | 1.062s | 0.861s |
| Median | 1.135s | 0.866s |
| p95 | 1.388s | 0.968s |

> MLC-LLM was ~19% faster than llama-cpp in this configuration.

---

## Notes

- **Qwen2.5-VL-7B-Instruct** requires ~16 GB VRAM and is not available on this device. The endpoint returns a descriptive stub response.
- **moondream2 via HuggingFace** downloads the model from `vikhyatk/moondream2` on first call — ensure internet access and that `transformers`, `torch`, and `pillow` are installed.
- **vLLM** follows the same proxy pattern as MLC-LLM — start the server natively, then point the FastAPI container at it via environment variable.
