import time
import os
import requests
from fastapi import FastAPI
from pydantic import BaseModel
from llama_cpp import Llama

app = FastAPI()

MLC_BASE_URL = os.getenv("MLC_BASE_URL", "http://host.docker.internal:8002")
MLC_MODEL = os.getenv("MLC_MODEL", "HF://mlc-ai/TinyLlama-1.1B-Chat-v1.0-q4f16_1-MLC")

llm = Llama(
    model_path="models/tinyllama-1.1b-chat-v1.0-q4_k_m.gguf",
    n_ctx=2048
)

class PromptRequest(BaseModel):
    prompt: str

@app.post("/generate")
def generate_text(request: PromptRequest):
    start_time = time.time()

    output = llm(
        request.prompt,
        max_tokens=100,
    )

    latency = time.time() - start_time

    return {
        "response": output["choices"][0]["text"],
        "latency_seconds": latency
    }

@app.post("/generate_mlc")
def generate_mlc(request: PromptRequest):
    start_time = time.time()

    payload = {
        "model": MLC_MODEL,
        "messages": [{"role": "user", "content": request.prompt}],
        "stream": False,
        "max_tokens": 100
    }

    r = requests.post(f"{MLC_BASE_URL}/v1/chat/completions", json=payload, timeout=300)
    r.raise_for_status()
    data = r.json()

    latency = time.time() - start_time
    text = data["choices"][0]["message"]["content"]

    return {
        "response": text,
        "latency_seconds": latency
    }
    