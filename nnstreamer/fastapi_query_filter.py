"""NNStreamer custom tensor_filter subplugin: 'phone -> FastAPI VLM server' bridge.

This is the actual NNStreamer plugin piece. NNStreamer's `tensor_filter` element
supports a `framework=python3` custom filter: you hand it a Python module with a
CustomFilter class, and it gets called once per frame inside the GStreamer
pipeline, same as a built-in inference backend would.

It can only be loaded inside a real NNStreamer + GStreamer runtime (the
`nnstreamer_python` module is provided by NNStreamer itself, not pip-installable),
which is not available on this machine (no official macOS package - see
../README.md). The actual network round-trip this class performs on every
`invoke()` is exactly what nnstreamer/query_client.py exercises standalone,
and that has been run and verified against the live server.

Pipeline usage (on a device with NNStreamer, e.g. Android/Tizen/Linux):

    gst-launch-1.0 \
        v4l2src ! videoconvert ! videoscale ! \
        video/x-raw,width=640,height=480,format=RGB ! jpegenc ! \
        tensor_converter ! \
        tensor_filter framework=python3 model=fastapi_query_filter.py ! \
        tensor_sink
"""
import sys

try:
    import nnstreamer_python as nns
except ImportError:
    nns = None  # only present inside an actual NNStreamer pipeline

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from vlm_bridge import DEFAULT_PROMPT, DEFAULT_SERVER_URL, query_vlm_server


class CustomFilter:
    """Registered with NNStreamer as: tensor_filter framework=python3 model=fastapi_query_filter.py"""

    def __init__(self, *args):
        # gst-launch ... tensor_filter framework=python3 model=fastapi_query_filter.py \
        #   custom=server_url:http://<laptop-ip>:8000/generate_hf,prompt:"What do you see?"
        self.server_url = DEFAULT_SERVER_URL
        self.prompt = DEFAULT_PROMPT
        for arg in args:
            if arg.startswith("server_url:"):
                self.server_url = arg.split(":", 1)[1]
            elif arg.startswith("prompt:"):
                self.prompt = arg.split(":", 1)[1]

    def getInputDim(self):
        # One JPEG-encoded frame per invoke, passed through as raw bytes.
        return nns.TensorShape([3, 480, 640, 1], nns.TensorType.UINT8)

    def getOutputDim(self):
        # Server's text response, UTF-8 bytes, fixed-size buffer.
        return nns.TensorShape([1024], nns.TensorType.UINT8)

    def invoke(self, input_array):
        frame_bytes = input_array[0].tobytes()
        try:
            result = query_vlm_server(frame_bytes, prompt=self.prompt, server_url=self.server_url)
            text = result.get("response", "")
        except Exception as e:
            text = f"[fastapi_query_filter error] {e}"

        encoded = text.encode("utf-8")[:1024].ljust(1024, b"\x00")
        return [encoded]
