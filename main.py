import time
import os
import base64
import requests
from io import BytesIO
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="LLM Server", description="Supports llama-cpp-python, MLC-LLM, and vLLM runtimes")

# ---------------------------------------------------------------------------
# Config (override via environment variables)
# ---------------------------------------------------------------------------
MLC_BASE_URL = os.getenv("MLC_BASE_URL", "http://host.docker.internal:8002")
MLC_MODEL = os.getenv("MLC_MODEL", "HF://mlc-ai/TinyLlama-1.1B-Chat-v1.0-q4f16_1-MLC")
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://host.docker.internal:8003")

LLAMA_CPP_TINYLLAMA_PATH = os.getenv(
    "LLAMA_CPP_TINYLLAMA_PATH",
    "models/tinyllama-1.1b-chat-v1.0-q4_k_m.gguf",
)
LLAMA_CPP_MOONDREAM_TEXT_PATH = os.getenv(
    "LLAMA_CPP_MOONDREAM_TEXT_PATH",
    "models/moondream2/moondream2-text-model-f16_ct-vicuna.gguf",
)

# ---------------------------------------------------------------------------
# Lazy-loaded model registry
# ---------------------------------------------------------------------------
_llama_cpp_tinyllama = None
_llama_cpp_moondream = None
_hf_moondream_model = None
_hf_moondream_tokenizer = None


def get_llama_cpp_tinyllama():
    global _llama_cpp_tinyllama
    if _llama_cpp_tinyllama is None:
        from llama_cpp import Llama
        _llama_cpp_tinyllama = Llama(model_path=LLAMA_CPP_TINYLLAMA_PATH, n_ctx=2048)
    return _llama_cpp_tinyllama


def get_llama_cpp_moondream():
    global _llama_cpp_moondream
    if _llama_cpp_moondream is None:
        from llama_cpp import Llama
        _llama_cpp_moondream = Llama(model_path=LLAMA_CPP_MOONDREAM_TEXT_PATH, n_ctx=2048)
    return _llama_cpp_moondream


def get_hf_moondream():
    """Load moondream2 from HuggingFace via the transformers library."""
    global _hf_moondream_model, _hf_moondream_tokenizer
    if _hf_moondream_model is None:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        revision = "2025-01-09"
        _hf_moondream_tokenizer = AutoTokenizer.from_pretrained(
            "vikhyatk/moondream2", revision=revision, trust_remote_code=True
        )
        _hf_moondream_model = AutoModelForCausalLM.from_pretrained(
            "vikhyatk/moondream2", revision=revision, trust_remote_code=True
        )
    return _hf_moondream_model, _hf_moondream_tokenizer


# ---------------------------------------------------------------------------
# Request schema
# ---------------------------------------------------------------------------
class PromptRequest(BaseModel):
    prompt: str
    # Supported runtimes: llama-cpp-python | mlc-llm | vllm
    runtime: str = "llama-cpp-python"
    # Supported models: tinyllama | moondream2 | qwen2.5-vl
    model: str = "tinyllama"
    # Base64-encoded image for VLMs (optional)
    image: Optional[str] = None
    max_tokens: int = 100


# ---------------------------------------------------------------------------
# Unified /generate endpoint — routes by runtime and model
# ---------------------------------------------------------------------------
@app.post("/generate")
def generate(request: PromptRequest):
    """
    Unified generation endpoint.

    - runtime: "llama-cpp-python" | "mlc-llm" | "vllm"
    - model:   "tinyllama" | "moondream2" | "qwen2.5-vl"
    - image:   base64-encoded image (optional, for VLMs via /generate_hf)
    """
    runtime = request.runtime.lower()

    if runtime == "llama-cpp-python":
        return _run_llama_cpp(request)
    elif runtime == "mlc-llm":
        return _run_mlc(request)
    elif runtime == "vllm":
        return _run_vllm(request)
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown runtime '{request.runtime}'. Choose from: llama-cpp-python, mlc-llm, vllm",
        )


# ---------------------------------------------------------------------------
# Runtime implementations
# ---------------------------------------------------------------------------
def _run_llama_cpp(request: PromptRequest) -> dict:
    model = request.model.lower()
    start = time.time()

    if model == "tinyllama":
        llm = get_llama_cpp_tinyllama()
        output = llm(request.prompt, max_tokens=request.max_tokens)
        text = output["choices"][0]["text"]

    elif model == "moondream2":
        # Text-model GGUF — text prompts only; for image VQA use /generate_hf
        llm = get_llama_cpp_moondream()
        output = llm(request.prompt, max_tokens=request.max_tokens)
        text = output["choices"][0]["text"]

    elif model == "qwen2.5-vl":
        return {
            "response": (
                "Qwen2.5-VL-7B-Instruct is not available on this device. "
                "It requires ~16 GB VRAM. "
                "To run locally: pip install transformers torch qwen-vl-utils, "
                "then load 'Qwen/Qwen2.5-VL-7B-Instruct' with transformers."
            ),
            "model": request.model,
            "runtime": "llama-cpp-python",
            "status": "unavailable",
        }

    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model '{request.model}' for llama-cpp-python. "
                   "Choose from: tinyllama, moondream2, qwen2.5-vl",
        )

    return {
        "response": text,
        "model": model,
        "runtime": "llama-cpp-python",
        "latency_seconds": round(time.time() - start, 4),
    }


def _run_mlc(request: PromptRequest) -> dict:
    start = time.time()
    payload = {
        "model": MLC_MODEL,
        "messages": [{"role": "user", "content": request.prompt}],
        "stream": False,
        "max_tokens": request.max_tokens,
    }
    try:
        r = requests.post(f"{MLC_BASE_URL}/v1/chat/completions", json=payload, timeout=300)
        r.raise_for_status()
    except requests.exceptions.ConnectionError:
        raise HTTPException(
            status_code=503,
            detail=f"MLC-LLM server not reachable at {MLC_BASE_URL}.",
        )
    data = r.json()
    return {
        "response": data["choices"][0]["message"]["content"],
        "model": request.model,
        "runtime": "mlc-llm",
        "latency_seconds": round(time.time() - start, 4),
    }


def _run_vllm(request: PromptRequest) -> dict:
    """
    Proxy to a vLLM OpenAI-compatible server.

    Start vLLM with:
        python -m vllm.entrypoints.openai.api_server \\
            --model <hf-model-id> --port 8003
    """
    start = time.time()
    payload = {
        "model": request.model,
        "messages": [{"role": "user", "content": request.prompt}],
        "max_tokens": request.max_tokens,
    }
    try:
        r = requests.post(f"{VLLM_BASE_URL}/v1/chat/completions", json=payload, timeout=300)
        r.raise_for_status()
    except requests.exceptions.ConnectionError:
        raise HTTPException(
            status_code=503,
            detail=(
                f"vLLM server not reachable at {VLLM_BASE_URL}. "
                f"Start it with: python -m vllm.entrypoints.openai.api_server "
                f"--model {request.model} --port 8003"
            ),
        )
    data = r.json()
    return {
        "response": data["choices"][0]["message"]["content"],
        "model": request.model,
        "runtime": "vllm",
        "latency_seconds": round(time.time() - start, 4),
    }


# ---------------------------------------------------------------------------
# /generate_hf — HuggingFace transformers VLM endpoint (new)
# ---------------------------------------------------------------------------
@app.post("/generate_hf")
def generate_hf(request: PromptRequest):
    """
    Run VLMs directly via HuggingFace transformers.

    Supported models:
      - moondream2  (vikhyatk/moondream2): pass image (base64) for visual Q&A,
                    or omit image for text-only generation.
      - qwen2.5-vl  (Qwen/Qwen2.5-VL-7B-Instruct): not available locally
                    (~16 GB VRAM required).
    """
    model = request.model.lower()
    start = time.time()

    if model == "moondream2":
        try:
            hf_model, tokenizer = get_hf_moondream()

            if request.image:
                from PIL import Image
                image_bytes = base64.b64decode(request.image)
                image = Image.open(BytesIO(image_bytes)).convert("RGB")
                enc_image = hf_model.encode_image(image)
                text = hf_model.answer_question(enc_image, request.prompt, tokenizer)
            else:
                inputs = tokenizer(request.prompt, return_tensors="pt")
                output_ids = hf_model.generate(**inputs, max_new_tokens=request.max_tokens)
                text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        except ImportError as e:
            raise HTTPException(
                status_code=500,
                detail=f"Missing dependency: {e}. Install with: pip install transformers torch pillow",
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

        return {
            "response": text,
            "model": "vikhyatk/moondream2",
            "runtime": "huggingface",
            "latency_seconds": round(time.time() - start, 4),
        }

    elif model == "qwen2.5-vl":
        return {
            "response": (
                "Qwen2.5-VL-7B-Instruct is not available on this device. "
                "It requires ~16 GB VRAM. "
                "To run: pip install transformers torch qwen-vl-utils accelerate, "
                "then load 'Qwen/Qwen2.5-VL-7B-Instruct' with transformers."
            ),
            "model": "Qwen/Qwen2.5-VL-7B-Instruct",
            "runtime": "huggingface",
            "status": "unavailable",
        }

    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model '{request.model}' for /generate_hf. Choose from: moondream2, qwen2.5-vl",
        )


# ---------------------------------------------------------------------------
# Legacy aliases — kept for backwards compatibility
# ---------------------------------------------------------------------------
@app.post("/generate_mlc")
def generate_mlc(request: PromptRequest):
    """Alias for /generate with runtime=mlc-llm."""
    request.runtime = "mlc-llm"
    return _run_mlc(request)


@app.post("/generate_moondream")
def generate_moondream(request: PromptRequest):
    """Alias for /generate with model=moondream2, runtime=llama-cpp-python."""
    request.model = "moondream2"
    request.runtime = "llama-cpp-python"
    return _run_llama_cpp(request)


@app.post("/generate_qwen")
def generate_qwen(request: PromptRequest):
    """Stub — Qwen2.5-VL-7B-Instruct is not available on this device."""
    return {
        "response": (
            "Qwen2.5-VL-7B-Instruct is not available on this device. "
            "It requires ~16 GB VRAM."
        ),
        "model": "Qwen/Qwen2.5-VL-7B-Instruct",
        "runtime": request.runtime,
        "status": "unavailable",
    }


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}
