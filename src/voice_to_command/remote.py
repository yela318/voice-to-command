"""Run whisper inference on a remote GPU box over plain HTTP.

The GPU box runs `v2c --serve --http`: it warms the model once, then answers
`POST /transcribe` with the recognised text. A thin client (`--server_ip` / the
`V2C_SERVER_IP` env var) never loads the model -- it ships the audio bytes over
and prints what comes back. stdlib only, no auth: meant for a trusted LAN, or an
SSH tunnel across anything less trusted.

Wire protocol:
  GET  /            -> 200 "ok"           (health check)
  POST /transcribe  -> 200 {"text": ...}  or  {"error": ...}
      header X-V2C-Format:    "file" (default) -- body is an audio file
                              "f32"            -- raw float32 mono @16 kHz
      header X-V2C-Translate: "1" translate to English, "0" keep the spoken
                              language, absent -> the server's own default
"""

from __future__ import annotations

import io
import json
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Union

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy is a hard dep, but keep import-safe
    np = None

DEFAULT_PORT = 8756

# Set this to a callable to receive every recognised sentence on the server as
# it is produced -- e.g. `from voice_to_command import remote; remote.on_text =
# planner.feed`. Called as on_text(text) after transcription, before the reply.
on_text = None


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802 - name fixed by BaseHTTPRequestHandler
        self._reply(200, b"ok\n", "text/plain")

    def do_POST(self):  # noqa: N802
        from .core import _log, transcribe

        t0 = time.perf_counter()
        fmt = self.headers.get("X-V2C-Format", "file")
        n = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(n)
        _log("recv", time.perf_counter() - t0, "({} KiB {})".format(n >> 10, fmt))

        xlate = self.headers.get("X-V2C-Translate")
        translate = None if xlate is None else xlate == "1"
        try:
            audio = np.frombuffer(body, dtype="<f4") if fmt == "f32" else io.BytesIO(body)
            text = transcribe(audio, translate=translate)  # logs model_load / infer / total
            print(text, flush=True)  # the recognised sentence, on the server's stdout
            if on_text is not None:
                on_text(text)
            payload = {"text": text}
        except Exception as exc:  # report, don't crash the server
            payload = {"error": "{}: {}".format(type(exc).__name__, exc)}
        self._reply(200, json.dumps(payload).encode("utf-8"), "application/json")
        _log("request", time.perf_counter() - t0)  # whole server-side handler

    def _reply(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args):  # quiet; core._log already goes to stderr
        pass


def serve_http(port: int = DEFAULT_PORT, host: str = None) -> int:
    """Warm the model, then answer /transcribe until Ctrl-C."""
    from .core import warmup

    host = host or os.environ.get("V2C_HTTP_HOST") or "0.0.0.0"
    httpd = ThreadingHTTPServer((host, port), _Handler)  # binds now -> fail fast
    warmup()
    print("ready ({}:{})".format(host, port), flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return 0


def transcribe_remote(
    audio: Union[str, "np.ndarray"],
    ip: str,
    port: int = DEFAULT_PORT,
    translate: bool = False,
    timeout: float = 120.0,
) -> str:
    """Send `audio` (a file path or float32 samples) to a `v2c --serve --http`
    box at `ip:port` and return the recognised text. `translate=True` asks the
    server to translate it to English.
    """
    import time
    import urllib.error
    import urllib.request

    from .core import _log

    if isinstance(audio, str):
        with open(audio, "rb") as fh:
            body, fmt = fh.read(), "file"
    else:
        body, fmt = np.asarray(audio, dtype="<f4").tobytes(), "f32"

    url = "http://{}:{}/transcribe".format(ip, port)
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/octet-stream",
            "X-V2C-Format": fmt,
            "X-V2C-Translate": "1" if translate else "0",
        },
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(
            "can't reach the v2c server at {} ({}). Start it on the GPU box with "
            "`v2c --serve --http`.".format(url, exc)
        ) from exc
    # client-side round trip: upload + server compute + download. The server logs
    # its own model_load/infer/total; the gap is network + (de)serialisation.
    _log("remote", time.perf_counter() - t0, "({} KiB {})".format(len(body) >> 10, fmt))
    if "error" in payload:
        raise RuntimeError("remote inference failed -- " + payload["error"])
    return payload["text"]
