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
  v4l2src ! videoconvert ! videoscale ! \
  video/x-raw,width=640,height=480,format=RGB ! jpegenc ! \
  tensor_converter ! \
  tensor_filter framework=python3 model=fastapi_query_filter.py \
    custom=server_url:http://<laptop-ip>:8000/generate_hf,prompt:"What do you see?" ! \
  tensor_sink
```

## What was actually tested, and what wasn't

**This Mac has no GStreamer or NNStreamer installed, and NNStreamer has no
official macOS package** (it officially targets Ubuntu, Tizen, Android, and
Yocto — confirmed via `brew search`, which has no `nnstreamer` formula). So
the literal `gst-launch-1.0` pipeline above has not been run here, and won't
run on this machine without an unofficial from-source build.

What *was* tested is the part that was actually uncertain: **whether the
FastAPI server correctly does image VLM inference given a JPEG frame**,
i.e. exactly what `fastapi_query_filter.invoke()` does on every frame. That
was run standalone via `query_client.py` against the live server on this
machine, using a synthetic 640x480 test frame in `sample_frames/`:

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

To run the *actual* NNStreamer pipeline, do the above on a Linux box (or
Android/Tizen device) with NNStreamer + its Python3 tensor_filter support
installed, point `v4l2src` at the phone's/device's camera, and set
`server_url` to the laptop's LAN IP.
