"""NNStreamer custom tensor_filter subplugin: 'phone -> FastAPI VLM server' bridge.

This is the actual NNStreamer plugin piece. NNStreamer's `tensor_filter` element
supports a `framework=python3` custom filter: you hand it a Python module with a
CustomFilter class, and it gets called once per frame inside the GStreamer
pipeline, same as a built-in inference backend would.

It receives raw RGB frames from `tensor_converter` (which doesn't accept
JPEG), JPEG-encodes each one, sends it to the FastAPI server through
vlm_bridge.py, and outputs the caption as a fixed-size UTF-8 byte tensor.

The `nnstreamer_python` module is provided by NNStreamer itself (not
pip-installable), so this only loads inside a real NNStreamer + GStreamer runtime.

Pipeline usage (on a device with NNStreamer, e.g. Android/Tizen/Linux):

    gst-launch-1.0 \
        v4l2src ! videoconvert ! videoscale ! videorate ! \
        video/x-raw,width=640,height=480,format=RGB,framerate=1/5 ! \
        tensor_converter ! \
        tensor_filter framework=python3 model=fastapi_query_filter.py \
            custom=server_url:http://<laptop-ip>:8000/generate_hf ! \
        tensor_sink
"""
import os
import sys
from io import BytesIO

import numpy as np
from PIL import Image

try:
    import nnstreamer_python as nns
except ImportError:
    nns = None  # only present inside an actual NNStreamer pipeline

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vlm_bridge import DEFAULT_PROMPT, DEFAULT_SERVER_URL, query_vlm_server

WIDTH, HEIGHT = 640, 480
OUTPUT_BYTES = 1024


def parse_custom(args):
    """Parse `custom=key:value,key:value` into a dict.

    Joined and re-split on commas so it works whether NNStreamer passes the
    custom string as one argument or splits it on whitespace.
    """
    options = {}
    for item in " ".join(args).split(","):
        key, sep, value = item.strip().partition(":")
        if sep:
            options[key] = value.strip().strip('"')
    return options


def encode_caption(text):
    """UTF-8 encode, truncate on a character boundary, and pad to OUTPUT_BYTES."""
    encoded = text.encode("utf-8")[:OUTPUT_BYTES].decode("utf-8", errors="ignore").encode("utf-8")
    return np.frombuffer(encoded.ljust(OUTPUT_BYTES, b"\x00"), dtype=np.uint8)


class CustomFilter:
    """Registered with NNStreamer as: tensor_filter framework=python3 model=fastapi_query_filter.py"""

    def __init__(self, *args):
        options = parse_custom(args)
        self.server_url = options.get("server_url", DEFAULT_SERVER_URL)
        self.prompt = options.get("prompt", DEFAULT_PROMPT)

    def setInputDim(self, input_dims):
        # Called by NNStreamer with the negotiated input; getInputDim/getOutputDim
        # leave invoke()'s input list empty on NNStreamer 2.7.
        # tensor_converter output for 640x480 RGB video: dims are channel:width:height:batch.
        dims = input_dims[0].getDims()[:3]
        if dims != [3, WIDTH, HEIGHT]:
            raise ValueError(f"expected 3:{WIDTH}:{HEIGHT} RGB frames, got {dims}")
        # Server's text response, UTF-8 bytes, fixed-size buffer.
        return [nns.TensorShape([OUTPUT_BYTES], np.uint8)]

    def invoke(self, input_array):
        frame = np.frombuffer(input_array[0].tobytes(), dtype=np.uint8).reshape(HEIGHT, WIDTH, 3)
        buf = BytesIO()
        Image.fromarray(frame, "RGB").save(buf, format="JPEG")
        try:
            result = query_vlm_server(buf.getvalue(), prompt=self.prompt, server_url=self.server_url)
            text = result.get("response", "")
        except Exception as e:
            text = f"[fastapi_query_filter error] {e}"
        print(f"[fastapi_query_filter] {text}", flush=True)
        return [encode_caption(text)]
