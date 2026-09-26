"""Autonomous Laya-driven Snake, streamed to LAN browsers over SSE.

The game runs server-side in its own thread: every move is a real Laya forward
pass on this machine. Browsers are pure spectators — they receive frames and
render them, and never drive the game.

    python -m snakeweb.server --host 0.0.0.0 --port 8080
"""

import argparse
import json
import queue
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .game import SnakeGame
from .policy import LayaPolicy

STATIC = Path(__file__).parent / "static"
MAX_CLIENTS = 32


class Broadcaster:
    """Fan out the newest frame to every connected spectator."""

    def __init__(self):
        self._lock = threading.Lock()
        self._subscribers = []
        self.latest = None
        # Set while at least one spectator is attached; the runner waits on this so
        # the Pi burns no CPU (and makes no Laya calls) when nobody is watching.
        self.has_clients = threading.Event()

    def subscribe(self):
        q = queue.Queue(maxsize=4)
        with self._lock:
            if len(self._subscribers) >= MAX_CLIENTS:
                return None
            self._subscribers.append(q)
            self.has_clients.set()
        return q

    def unsubscribe(self, q):
        with self._lock:
            if q in self._subscribers:
                self._subscribers.remove(q)
            if not self._subscribers:
                self.has_clients.clear()

    def publish(self, frame):
        self.latest = frame
        with self._lock:
            targets = list(self._subscribers)
        for q in targets:
            try:
                q.put_nowait(frame)
            except queue.Full:
                # A slow client falls behind rather than stalling the game loop.
                pass

    @property
    def client_count(self):
        with self._lock:
            return len(self._subscribers)


class Runner(threading.Thread):
    """Plays the game forever, one Laya decision per tick."""

    daemon = True

    def __init__(self, broadcaster, *, width, height, seed, guarded, min_tick):
        super().__init__(name="snake-runner")
        self.bus = broadcaster
        self.width, self.height, self.seed = width, height, seed
        self.guarded, self.min_tick = guarded, min_tick
        self.policy = None
        self.round = 1
        self.best = 0
        self.deaths = 0
        self.interventions = 0
        self.started = time.time()
        self.ready = threading.Event()

    def _frame(self, game, decision, *, status):
        elapsed = time.time() - self.started
        moves_done = max(1, self.moves_total)
        return {
            "game": game.snapshot(),
            "decision": decision.to_dict() if decision else {},
            "stats": {
                "round": self.round,
                "best": self.best,
                "deaths": self.deaths,
                "interventions": self.interventions,
                "guarded": self.guarded,
                "elapsed": elapsed,
                "steps_per_second": moves_done / max(0.001, elapsed),
                "clients": self.bus.client_count,
                "status": status,
                **self.policy.metadata,
            },
        }

    def run(self):
        self.moves_total = 0
        self.policy = LayaPolicy(guarded=self.guarded)
        self.ready.set()

        while True:
            # Idle until a player attaches: no game churns, no Laya calls, while the
            # model just sits resident in the laya-serve container.
            self.bus.has_clients.wait()

            game = SnakeGame(width=self.width, height=self.height, seed=self.seed + self.round)
            self.bus.publish(self._frame(game, None, status="LIVE"))

            while game.alive and not game.won:
                # Pause mid-round if the last spectator leaves; resume when one returns.
                if not self.bus.has_clients.is_set():
                    self.bus.publish(self._frame(game, None, status="PAUSED · no player"))
                    self.bus.has_clients.wait()
                tick_start = time.perf_counter()
                try:
                    decision = self.policy.decide(game)
                except RuntimeError:
                    # Trapped with the shield on: end the round rather than crash.
                    game.alive, game.death_reason = False, "trapped"
                    break
                if decision.intervened:
                    self.interventions += 1
                game.step(decision.executed)
                self.moves_total += 1
                self.best = max(self.best, game.score)
                status = "BOARD CLEAR" if game.won else "GAME OVER" if not game.alive else "LIVE"
                self.bus.publish(self._frame(game, decision, status=status))
                remaining = self.min_tick - (time.perf_counter() - tick_start)
                if remaining > 0:
                    time.sleep(remaining)

            if not game.won:
                self.deaths += 1
            self.bus.publish(
                self._frame(
                    game, None, status="BOARD CLEAR" if game.won else "GAME OVER"
                )
            )
            time.sleep(3)
            self.round += 1


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    bus: Broadcaster = None

    def log_message(self, fmt, *args):
        pass  # Keep the console readable; the game loop prints its own progress.

    def _send(self, code, body, content_type):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/":
            self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
        elif path == "/api/state":
            payload = json.dumps(self.bus.latest or {}).encode()
            self._send(200, payload, "application/json")
        elif path == "/events":
            self._stream()
        elif self._serve_static(path):
            pass
        else:
            self._send(404, b"not found", "text/plain")

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

    def _stream(self):
        q = self.bus.subscribe()
        if q is None:
            self._send(503, b"too many spectators", "text/plain")
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        try:
            if self.bus.latest:
                self._event(self.bus.latest)
            while True:
                try:
                    self._event(q.get(timeout=15))
                except queue.Empty:
                    self.wfile.write(b": keepalive\n\n")  # Hold idle proxies open.
                    self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            self.bus.unsubscribe(q)

    def _event(self, frame):
        self.wfile.write(f"data: {json.dumps(frame)}\n\n".encode())
        self.wfile.flush()


def lan_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))  # No packets sent; just resolves the outbound interface.
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--width", type=int, default=24)
    ap.add_argument("--height", type=int, default=16)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--min-tick", type=float, default=0.0, help="Floor on seconds per move")
    ap.add_argument("--no-shield", action="store_true", help="Disable the cycle safety shield")
    args = ap.parse_args()

    bus = Broadcaster()
    runner = Runner(
        bus,
        width=args.width,
        height=args.height,
        seed=args.seed,
        guarded=not args.no_shield,
        min_tick=args.min_tick,
    )
    from .policy import LAYA_URL

    print(f"waiting for laya-serve at {LAYA_URL} (it preloads checkpoints on first boot)...")
    runner.start()
    runner.ready.wait()
    print("laya-serve reachable; game running")

    Handler.bus = bus
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    print(f"spectate on the LAN:  http://{lan_ip()}:{args.port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")


if __name__ == "__main__":
    main()
