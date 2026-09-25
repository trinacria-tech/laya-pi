"""A small web server showcasing the Laya decision API with five example decisions.

Serves a page of example cards; each "Run" posts the example's {state, questions}
to the Laya API and shows the typed answers. Inference calls are proxied
server-side (browser -> this server -> LAYA_URL), so the page stays same-origin
and needs no CORS on the API.

    LAYA_URL=http://laya-onnx:8001 python -m showcase.server --port 8090
"""

import argparse
import json
import os
import socket
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .examples import EXAMPLES

STATIC = Path(__file__).parent / "static"
LAYA_URL = os.environ.get("LAYA_URL", "http://localhost:8001").rstrip("/")
TIMEOUT = float(os.environ.get("LAYA_TIMEOUT", "60"))
MAX_BODY = 64 * 1024

BY_ID = {e["id"]: e for e in EXAMPLES}


def ask_laya(state, questions):
    payload = json.dumps({"state": state, "questions": questions}).encode()
    req = urllib.request.Request(
        f"{LAYA_URL}/v1/systemone",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        return json.loads(resp.read())


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def _send(self, code, body, ctype):
        if isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/":
            self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
        elif path == "/api/examples":
            # The catalogue, minus nothing -- the page renders questions client-side.
            self._send(200, json.dumps(EXAMPLES), "application/json")
        elif self._serve_static(path):
            pass
        else:
            self._send(404, "not found", "text/plain")

    _CTYPES = {".ttf": "font/ttf", ".woff2": "font/woff2", ".css": "text/css",
               ".js": "text/javascript", ".png": "image/png", ".svg": "image/svg+xml"}

    def _serve_static(self, path):
        """Serve a file under static/, e.g. /fonts/*.ttf. Rejects path traversal."""
        target = (STATIC / path.lstrip("/")).resolve()
        if not str(target).startswith(str(STATIC.resolve()) + "/") or not target.is_file():
            return False
        ctype = self._CTYPES.get(target.suffix, "application/octet-stream")
        self._send(200, target.read_bytes(), ctype)
        return True

    def do_POST(self):
        if self.path.split("?")[0] != "/api/run":
            self._send(404, "not found", "text/plain")
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            if length > MAX_BODY:
                self._send(413, json.dumps({"error": "body too large"}), "application/json")
                return
            body = json.loads(self.rfile.read(length))
        except (ValueError, TypeError):
            self._send(400, json.dumps({"error": "invalid JSON"}), "application/json")
            return

        # Either an example id (run the curated one) or an ad-hoc {state, questions}.
        if "id" in body:
            example = BY_ID.get(body["id"])
            if not example:
                self._send(404, json.dumps({"error": "unknown example"}), "application/json")
                return
            state = body.get("state", example["state"])  # allow the user to edit the text
            questions = example["questions"]
        else:
            state, questions = body.get("state"), body.get("questions")
            if not questions:
                self._send(400, json.dumps({"error": "missing questions"}), "application/json")
                return

        try:
            result = ask_laya(state, questions)
        except (urllib.error.URLError, TimeoutError) as e:
            self._send(502, json.dumps({"error": f"laya unreachable: {e}"}), "application/json")
            return
        self._send(200, json.dumps(result), "application/json")


def lan_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8090)
    args = ap.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    print(f"showcase on the LAN:  http://{lan_ip()}:{args.port}/   (api: {LAYA_URL})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")


if __name__ == "__main__":
    main()
