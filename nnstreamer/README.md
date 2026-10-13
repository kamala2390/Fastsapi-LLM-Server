# NNStreamer bridge: mobile phone -> FastAPI VLM server

Target example chosen: **a phone captures a camera frame, an NNStreamer
pipeline on the phone sends it to this repo's FastAPI server, and the server's
moondream2 VLM sends back a text description of the frame.**

## Why this example, and which endpoint

The server has several endpoints (see the top-level README), but only
`POST /generate_hf` (model=`moondream2`) does real image inference — it's the
only one where the "vision" half of a VLM pipeline actually exists today.
Request/response contract:

```
POST http://<laptop-ip>:8000/generate_hf
Content-Type: application/json

{
  "prompt": "What do you see in this image?",
  "model": "moondream2",
  "image": "<base64-encoded JPEG/PNG bytes>",
  "max_tokens": 100
}
```
```json
{
  "response": "<model's answer>",
  "model": "vikhyatk/moondream2",
  "runtime": "huggingface",
  "latency_seconds": 5.66
}
```
No auth, no CORS restriction — a phone on the same LAN can hit this directly.

## Why this isn't a stock NNStreamer element

NNStreamer's built-in `tensor_query_client`/`tensor_query_server` pair
already does "phone offloads inference to a bigger machine over the network",
but it speaks NNStreamer's own edge wire protocol, not plain JSON+base64 REST.
Since this server only speaks REST, the "plugin" here is a custom NNStreamer
`tensor_filter` **subplugin** (NNStreamer's supported way to add a new
inference backend into a pipeline) that does the JSON/base64 HTTP call in its
`invoke()`:

- [`fastapi_query_filter.py`](fastapi_query_filter.py) — the actual plugin.
  Registered into a pipeline as `tensor_filter framework=python3
  model=fastapi_query_filter.py`. Runs once per frame.
- [`vlm_bridge.py`](vlm_bridge.py) — the HTTP call itself (encode frame ->
  base64 -> POST -> parse JSON), shared so it can be exercised outside NNStreamer.
- [`query_client.py`](query_client.py) — standalone CLI using the same bridge,
  standing in for the phone side, for testing without NNStreamer installed.

Full pipeline, as it would run on the phone (Android/Tizen/Linux with
NNStreamer installed):

```bash
gst-launch-1.0 \
  v4l2src ! videoconvert ! videoscale ! videorate ! \
  video/x-raw,width=640,height=480,format=RGB,framerate=1/5 ! \
  tensor_converter ! \
  tensor_filter framework=python3 model=fastapi_query_filter.py \
    custom=server_url:http://<laptop-ip>:8000/generate_hf,prompt:"What do you see?" ! \
  tensor_sink
```

Notes on the pipeline:
- `tensor_converter` doesn't accept JPEG, so raw 640x480 RGB frames
  (tensor dims `3:640:480:1`) go into the filter, and the filter
  JPEG-encodes each frame before POSTing it.
- Each server call takes ~5-16 s, so `videorate ... framerate=1/5` limits the
  camera to one frame every 5 seconds instead of ~30 fps, which would stall
  the pipeline.
- The caption comes out as a 1024-byte UTF-8 tensor and is also printed to
  stdout.

## What was tested

### 1. Real NNStreamer pipeline (Ubuntu container -> server on the Mac)

NNStreamer has no official macOS package (it targets Ubuntu, Tizen, Android
and Yocto), so the pipeline was run in an Ubuntu 22.04 container with the
official NNStreamer 2.7 packages ([`Dockerfile`](Dockerfile)), talking to the
FastAPI server running natively on the Mac. A still test image stands in for
the camera (`filesrc ... ! imagefreeze`), since no phone camera is attached;
everything from `tensor_converter` on is the same as the phone pipeline:

```
$ docker build -t nns-test nnstreamer
$ docker run --rm -v "$PWD/nnstreamer:/work" nns-test \
    gst-launch-1.0 -q \
      filesrc location=sample_frames/test_frame.jpg ! jpegdec ! videoconvert ! videoscale ! imagefreeze num-buffers=2 ! \
      video/x-raw,width=640,height=480,format=RGB,framerate=1/5 ! \
      tensor_converter ! \
      tensor_filter framework=python3 model=/work/fastapi_query_filter.py \
        custom=server_url:http://host.docker.internal:8000/generate_hf,prompt:"Describe this image in one sentence." ! \
      tensor_sink

[fastapi_query_filter] A yellow circle with a brown triangle sits atop a green square, creating a balanced composition against a light blue background.
[fastapi_query_filter] A yellow circle with a brown triangle sits atop a green square, creating a balanced composition against a light blue background.
```

Both frames were captioned correctly and the pipeline exited with status 0
(server log: two `POST /generate_hf 200 OK`). NNStreamer prints a harmless
`Given input info is invalid, cannot get output info` warning during caps
negotiation before the frames flow.

Things learned from running it in real NNStreamer (2.7):
- Declaring fixed shapes with `getInputDim`/`getOutputDim` made `invoke()`
  receive an empty input list, so the filter uses `setInputDim` instead.
  Each frame arrives as one flat 921600-byte (640x480x3) uint8 array.
- NNStreamer splits the `custom=` string on spaces, so the filter re-joins
  it before parsing `key:value` pairs.

The same pipeline was also run against the server inside its Docker
container (`http://host.docker.internal:8001/generate_hf`) and returned
"The image contains a square, a circle, and a triangle." Docker on a Mac has
no GPU access, so moondream2 runs on CPU there: ~47 s per frame, vs ~6 s when
the server runs natively on Apple Silicon (MPS).

Not tested yet: an actual phone camera (`v4l2src` / Android camera source)
over Wi-Fi.

### 2. Server round trip via the standalone client

`query_client.py` sends the test frame through the same bridge code against
the live server on the Mac, using a synthetic 640x480 test frame in
`sample_frames/`:

```
$ python3 query_client.py sample_frames/test_frame.jpg "What do you see in this image?"
-> POST http://localhost:8000/generate_hf
-> prompt: 'What do you see in this image?'
-> frame: sample_frames/test_frame.jpg (9894 bytes)

<- model: vikhyatk/moondream2
<- server latency_seconds: 16.5429
<- round-trip wall time: 16.551s
<- response: The image features a simple geometric design with a yellow circle and a brown triangle. The yellow circle is positioned above a green square, creating a visually appealing composition.
```

```
$ python3 query_client.py sample_frames/test_frame.jpg
-> POST http://localhost:8000/generate_hf
-> prompt: 'Describe what you see in this image in one sentence.'
-> frame: sample_frames/test_frame.jpg (9894 bytes)

<- model: vikhyatk/moondream2
<- server latency_seconds: 5.6634
<- round-trip wall time: 5.677s
<- response: A yellow circle with a brown triangle on top is centered on a green square, creating a balanced composition against a light blue background.
```

(`sample_frames/test_frame.jpg` is a synthetic sun/triangle/rectangle test
image, since no phone camera is attached in this environment — moondream2
correctly described its shapes and colors both times, confirming the
image-in/text-out round trip works.)

The first call is slower because it's also the model's first invocation
(weight materialization); subsequent calls only pay inference cost.

## Bug found and fixed while testing

Getting even the server round-trip working surfaced a real, pre-existing bug:
`main.py`'s `get_hf_moondream()` never moved the model off the CPU, but
moondream2's own remote code (`vision.py`) assumes that if MPS (Apple Silicon
GPU) is available, the model must live there — it hardcodes `.to("mps")` in
a pooling workaround regardless of where the rest of the model actually is.
On this Apple Silicon machine that produced `Passed CPU tensor to MPS op`.
Fixed by moving the model to `mps` explicitly when available, matching what
moondream2's own code already assumes (see the `get_hf_moondream` diff in
this PR).

## Running it yourself

```bash
# one-time setup (this Mac needed these on top of requirements.txt):
pip install pillow einops pyvips   # + `brew install vips` for libvips itself

# start the server
uvicorn main:app --host 0.0.0.0 --port 8000

# from the phone's machine, or simulated here on the same machine:
cd nnstreamer
python3 query_client.py sample_frames/test_frame.jpg "What do you see?"
```

To run the NNStreamer pipeline on a Mac, use the Ubuntu container in
section 1. On a Linux box (or Android/Tizen device) with NNStreamer + its
Python3 tensor_filter support installed, use the phone pipeline above with
`v4l2src` pointed at the device's camera and `server_url` set to the laptop's
LAN IP.
