# Step-by-step guide

How to run the FastAPI LLM server, test the LLM and VLM models on each
runtime, and run the NNStreamer pipeline that sends camera frames to the
server. Commands are for macOS (Apple Silicon) unless noted.

---

## 1. Concepts

### Models vs. runtimes
Every request picks a **model** (the AI) and a **runtime** (the engine that runs it).

| Model | What it is | Images? |
|---|---|---|
| `tinyllama` | Small text chatbot (1.1B params, 4-bit GGUF, ~670 MB) | No |
| `moondream2` | Small vision-language model (VLM) | Yes, via the HuggingFace runtime only |
| `qwen2.5-vl` | Larger VLM (7B). Needs ~16 GB GPU memory, so it's a stub that returns `"status": "unavailable"` | — |

| Runtime | What it is | How it runs here |
|---|---|---|
| `llama-cpp-python` (default) | Python wrapper for llama.cpp, a C++ engine for `.gguf` models | Inside the server, using files in `models/` |
| `mlc-llm` | Engine compiled for specific hardware (e.g. Mac GPU via Metal) | Separate server on port 8002; this server forwards to it |
| `vllm` | High-throughput engine for NVIDIA GPU servers | Separate server on port 8003; forwarding only |
| `huggingface` | The `transformers` library | Inside the server; the **only runtime that processes images** (moondream2) |

moondream2 exists twice: a `.gguf` copy (llama.cpp, text only) and a
HuggingFace copy (can see images). Any request that includes an `image` is
automatically routed to the HuggingFace copy.

### Why NNStreamer
[GStreamer](https://gstreamer.freedesktop.org/) is a framework for media
pipelines: chains of small elements that process video or audio frame by
frame (camera → convert → resize → ...).
[NNStreamer](https://nnstreamer.ai/) adds neural-network elements to those
pipelines (`tensor_converter`, `tensor_filter`, `tensor_sink`). It's the
standard way to run AI on camera streams on edge devices: Android, Tizen and
embedded Linux boards.

An edge device like a phone has a camera but is too weak to run a VLM well.
The laptop has the model but no camera. This project connects them:

```
PHONE / EDGE DEVICE (NNStreamer pipeline)              LAPTOP (FastAPI server)
camera → convert → tensor_converter → tensor_filter ──HTTP──► /generate_hf → moondream2
                                          ▲                                    │
                                          └────────── caption text ────────────┘
```

`tensor_filter` normally runs a model on the device. Our plugin,
[`nnstreamer/fastapi_query_filter.py`](nnstreamer/fastapi_query_filter.py),
replaces that with "send this frame to the FastAPI server and return its
answer". This edge-to-server setup is the starting point for studying how
LLMs on embedded systems behave under attacks and faults.

The server knows nothing about NNStreamer; it just receives an HTTP request
with a base64 image.

---

## 2. Prerequisites

- **Docker Desktop**, running (whale icon in the menu bar)
- **~25 GB free disk.** The server image is ~11 GB. Check with `df -h /`
  and free Docker space with `docker system prune -a`.
- **Model files** in `models/` (not in git):
  - `models/tinyllama-1.1b-chat-v1.0-q4_k_m.gguf`
  - `models/moondream2/moondream2-text-model-f16_ct-vicuna.gguf`
- **Python venv** (only for running natively or using `query_client.py`):
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  pip install -r requirements.txt
  brew install vips
  ```

All commands below are run from the repo root (`fastapi-llm-server/`)
unless stated otherwise.

> **VS Code terminal tip:** VS Code auto-types `source venv/bin/activate`
> into new terminals. If you start a command before that finishes, the
> command gets cancelled (`Canceled: context canceled`). Wait for `(venv)` to
> show in the prompt first.

---

## 3. Start the server

Pick **one** of the two options.

### Option A: Docker (portable, CPU only, slower)

```bash
docker build -t llm-server .
docker run -p 8001:8000 -v ~/.cache/huggingface:/root/.cache/huggingface llm-server
```

- `-p 8001:8000` exposes the container's port 8000 on the Mac as **8001**.
- `-v ~/.cache/huggingface:...` shares downloaded moondream2 weights so the
  container doesn't re-download 3.6 GB.
- Ready when you see `Uvicorn running on http://0.0.0.0:8000`.
- Docker on a Mac can't use the Apple GPU, so image requests take **~45–50 s**
  (the first one can take 1–2 min while the model loads).

### Option B: Native (uses the Mac GPU, faster)

```bash
source venv/bin/activate
uvicorn main:app --host 0.0.0.0 --port 8001
```

Image requests take **~6 s**.

### Check it's up
```bash
curl localhost:8001/health
# {"status":"ok"}
```

---

## 4. Open the FastAPI docs page

Open **http://localhost:8001/docs** in a browser. This is FastAPI's built-in
Swagger UI. For each endpoint: click it → **Try it out** → paste a JSON body →
**Execute**.

Opening an endpoint URL like `localhost:8001/generate_hf` directly in the
browser returns `Method Not Allowed`. That's expected: browsers send GET, and
these endpoints only accept POST.

### Endpoints

| Endpoint | Purpose |
|---|---|
| `GET /health` | Health check |
| `POST /generate` | Main endpoint: choose `runtime` and `model`. Requests with an `image` go to moondream2 (HuggingFace) |
| `POST /generate_hf` | moondream2 image Q&A via HuggingFace (image required) |
| `POST /generate_mlc` | Shortcut for runtime `mlc-llm` |
| `POST /generate_moondream` | moondream2 text via llama.cpp; with an `image`, routes to `/generate_hf` |
| `POST /generate_qwen` | Qwen2.5-VL stub |

Request fields: `prompt` (required), `runtime`, `model`, `image` (base64),
`max_tokens` (default 100).

---

## 5. Test the LLMs (text)

### llama-cpp-python + TinyLlama
`POST /generate`
```json
{"prompt": "The capital of France is", "runtime": "llama-cpp-python", "model": "tinyllama", "max_tokens": 20}
```
Expect a text response, `"runtime": "llama-cpp-python"`, ~1 s.

### llama-cpp-python + moondream2 (text only)
`POST /generate_moondream`
```json
{"prompt": "Q: What is the capital of France?\nA:", "max_tokens": 20}
```
Expect `" Paris"`.

### MLC-LLM + TinyLlama
MLC must run **natively** in a separate terminal (Docker can't use Metal):
```bash
pip install mlc-llm   # see https://llm.mlc.ai for the right install command
python3 -m mlc_llm serve HF://mlc-ai/TinyLlama-1.1B-Chat-v1.0-q4f16_1-MLC --device metal --port 8002
```
Make sure the server knows where MLC is:
- Docker: `docker run -p 8001:8000 -e MLC_BASE_URL=http://host.docker.internal:8002 ... llm-server` (this is the default)
- Native: `MLC_BASE_URL=http://localhost:8002 uvicorn main:app --host 0.0.0.0 --port 8001`

Then `POST /generate`:
```json
{"prompt": "The capital of France is", "runtime": "mlc-llm", "max_tokens": 20}
```
Without the MLC server running, you get a **503 "not reachable"** error,
which is correct behavior.

### vLLM
vLLM needs an NVIDIA GPU, so it won't run on a Mac. On a GPU machine:
```bash
python -m vllm.entrypoints.openai.api_server --model <hf-model-id> --port 8003
```
Then `POST /generate` with `"runtime": "vllm", "model": "<hf-model-id>"`.
Without it you get a 503, which is expected.

### Qwen2.5-VL stub
`POST /generate_qwen` with `{"prompt": "hi"}` → `"status": "unavailable"`.

### Compare runtime latency
```bash
python3 bench.py
```
Runs 3 warmup and 20 timed requests against llama-cpp and MLC and prints
average, median and p95 latency. It targets `localhost:8000`; edit the URLs
in `bench.py` if your server is on 8001.

---

## 6. Test the VLM (images)

### Easiest: `query_client.py`
```bash
cd nnstreamer
python3 query_client.py sample_frames/test_frame.jpg "Describe this image." http://localhost:8001/generate_hf
cd ..
```
Expected (test image = yellow circle, brown triangle, green square):
```
<- model: vikhyatk/moondream2
<- server latency_seconds: ~6 (native) / ~50 (Docker)
<- response: A yellow circle with a brown triangle sits atop a green square...
```
Works with any JPG or PNG path on your Mac.

### In the /docs page
Copy an image as base64 to the clipboard:
```bash
base64 -i nnstreamer/sample_frames/test_frame.jpg | pbcopy
```
Then paste it into `"image"`:

| Endpoint | Body | Expected |
|---|---|---|
| `POST /generate_hf` | `{"prompt": "Describe this image.", "model": "moondream2", "image": "<paste>"}` | Description, `"runtime": "huggingface"` |
| `POST /generate` | same | same (auto-routed to HuggingFace) |
| `POST /generate_moondream` | same | same |

### Error checks (instant)

| Endpoint | Body | Expected |
|---|---|---|
| `POST /generate_hf` | `{"prompt": "hi", "model": "moondream2"}` | 400: requires an image |
| `POST /generate` | `{"prompt": "hi", "model": "tinyllama", "image": "<paste>"}` | 400: model does not accept images |
| `POST /generate` | `{"prompt": "hi", "runtime": "banana"}` | 400: unknown runtime |

---

## 7. Run the NNStreamer pipeline

NNStreamer has no macOS package, so it runs in an Ubuntu container
([`nnstreamer/Dockerfile`](nnstreamer/Dockerfile)) that plays the role of the
phone. Keep the server from step 3 running.

### 7.1 Build the NNStreamer image (one time)
```bash
docker build -t nns-test nnstreamer
```

### 7.2 Check NNStreamer is installed (optional)
```bash
docker run --rm nns-test gst-inspect-1.0 tensor_filter | grep "Available frameworks"
# ... {custom, custom-easy, cpp, python3}
```

### 7.3 Run the pipeline
```bash
docker run --rm -v "$PWD/nnstreamer:/work" nns-test \
  gst-launch-1.0 -q \
    filesrc location=sample_frames/test_frame.jpg ! decodebin ! videoconvert ! videoscale ! imagefreeze num-buffers=1 ! \
    video/x-raw,width=640,height=480,format=RGB,framerate=1/5 ! \
    tensor_converter ! \
    tensor_filter framework=python3 model=/work/fastapi_query_filter.py \
      custom=server_url:http://host.docker.internal:8001/generate_hf,prompt:"What shapes are in this image?" ! \
    tensor_sink
```

| Part | Tool | Job |
|---|---|---|
| `filesrc ... ! decodebin` | GStreamer | Read and decode the image (stands in for the camera) |
| `videoconvert ! videoscale ! imagefreeze` | GStreamer | Turn it into 640x480 RGB video frames |
| `tensor_converter` | NNStreamer | Frame → tensor |
| `tensor_filter ... fastapi_query_filter.py` | NNStreamer + our plugin | Send the frame to the server, get the caption |
| `custom=server_url:...,prompt:...` | plugin settings | Server address (`host.docker.internal` = the Mac) and question |
| `tensor_sink` | NNStreamer | End of pipeline |

### 7.4 Expected output
```
WARNING **: Given input info is invalid, cannot get output info.   ← harmless
[fastapi_query_filter] The image contains a square, a circle, and a triangle.
```
The server log shows `"POST /generate_hf HTTP/1.1" 200 OK`.

### 7.5 Use your own image
1. Put it **inside `nnstreamer/`** (the container can only see that folder):
   ```bash
   cp ~/Downloads/photo.jpg nnstreamer/sample_frames/
   ```
2. Check it's a real image:
   ```bash
   file nnstreamer/sample_frames/photo.jpg
   # should say "JPEG image data" or "PNG image data", not "HTML document"
   ```
3. Change `location=sample_frames/photo.jpg` in the command above.

Notes:
- Convert iPhone HEIC first: `sips -s format jpeg in.HEIC --out nnstreamer/sample_frames/photo.jpg`
- Shrink very large photos: `sips -Z 1024 big.jpg --out nnstreamer/sample_frames/photo.jpg`
- Don't put commas in the prompt; the plugin uses commas to separate settings.
- `num-buffers=3` sends 3 frames and prints 3 captions.

### 7.6 On a real device
On Linux/Android/Tizen with NNStreamer and its Python3 filter installed,
replace the file source with the camera and point at the laptop's LAN IP
(find it with `ipconfig getifaddr en0` on the Mac):
```bash
gst-launch-1.0 \
  v4l2src ! videoconvert ! videoscale ! videorate ! \
  video/x-raw,width=640,height=480,format=RGB,framerate=1/5 ! \
  tensor_converter ! \
  tensor_filter framework=python3 model=fastapi_query_filter.py \
    custom=server_url:http://<laptop-ip>:8001/generate_hf,prompt:"What do you see?" ! \
  tensor_sink
```
`framerate=1/5` sends one frame every 5 s, because each server call takes
seconds and the camera would otherwise produce ~30 frames/s.

---

## 8. Troubleshooting

| Error | Cause / fix |
|---|---|
| `Canceled: context canceled` during `docker build` | VS Code interrupted it while activating the venv. Wait for `(venv)` and rerun. |
| `Cannot connect to the Docker daemon` | Start Docker Desktop. |
| `input/output error` or `no space left` during build | Disk full. `docker system prune -a`, then `docker builder prune -a`. |
| `Bind for 0.0.0.0:8001 failed: port is already allocated` | A server is already running on 8001. Use it, or `docker ps` → `docker stop <name>`. |
| `Method Not Allowed` in the browser | Expected for POST endpoints; use `/docs`. |
| `query_client.py` prints the request and waits | Normal: Docker takes ~50 s, up to 2 min for the first request. |
| `ReadTimeout` | First request too slow while the model loads. Run it again. |
| `path "nnstreamer" not found` | You're inside `nnstreamer/`. `cd ..` to the repo root. |
| `No valid frames decoded before end of stream` | The file isn't a real image (check with `file`), or `jpegdec` was used on a progressive JPEG. Use `decodebin`. |
| `Connection refused` from the plugin | Server isn't running, or the port in `server_url` is wrong. |
| 503 "not reachable" for `mlc-llm` / `vllm` | That external server isn't running (see step 5). |

---

## 9. Stop and clean up

```bash
# stop the server container
docker stop $(docker ps -q --filter ancestor=llm-server)

# stop a native server
pkill -f "uvicorn main:app"

# free Docker disk space (removes unused images and build cache)
docker system prune -a
```
