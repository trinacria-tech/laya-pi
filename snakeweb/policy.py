"""Snake decisions delegated to a laya-serve instance over HTTP.

The model lives in its own container (`laya-serve`, Laya's own FastAPI server on
the Jev `/v1/systemone` wire protocol). This process holds no weights: it builds
the planner features, asks the server, and applies the deterministic safety
shield to whatever comes back.

Adapted from laya-mlx (Apache-2.0); see NOTICE.
"""

import json
import math
import os
import platform
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass

from .game import DIRECTIONS

LAYA_URL = os.environ.get("LAYA_URL", "http://localhost:8000")
LAYA_MODEL = os.environ.get("LAYA_MODEL", "multilingual")
LAYA_API_KEY = os.environ.get("LAYA_API_KEY")
TIMEOUT = float(os.environ.get("LAYA_TIMEOUT", "120"))


class LayaUnavailable(RuntimeError):
    pass


@dataclass
class Decision:
    probabilities: dict
    proposed: str
    executed: str
    safe_directions: list
    intervened: bool
    dead_end_risk: float
    food_reachable: float
    inference_ms: float
    decision_ms: float
    input_tokens: int
    output_tokens: int
    safe_count: int
    planner_best: str

    def to_dict(self):
        return asdict(self)


def _probabilities(answer, directions):
    """laya-serve returns a full distribution for choice answers; fall back if absent."""
    probs = answer.get("probabilities")
    if isinstance(probs, dict) and probs:
        return {d: float(probs.get(d, 0.0)) for d in directions}
    chosen = answer.get("choice")
    confidence = float(answer.get("confidence", 1.0))
    rest = (1.0 - confidence) / max(1, len(directions) - 1)
    return {d: (confidence if d == chosen else rest) for d in directions}


class LayaClient:
    def __init__(self, base_url=LAYA_URL, *, model=LAYA_MODEL, api_key=LAYA_API_KEY):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key

    def _post(self, path, payload):
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        if self.api_key:
            request.add_header("Authorization", f"Bearer {self.api_key}")
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as error:
            raise LayaUnavailable(f"laya-serve {error.code}: {error.read()[:200]!r}") from error
        except (urllib.error.URLError, TimeoutError) as error:
            raise LayaUnavailable(f"laya-serve unreachable at {self.base_url}: {error}") from error

    def health(self):
        try:
            with urllib.request.urlopen(f"{self.base_url}/health", timeout=5) as response:
                return json.loads(response.read())
        except Exception as error:
            raise LayaUnavailable(f"laya-serve not healthy: {error}") from error

    def wait_until_ready(self, attempts=120, delay=5):
        """Preloading three checkpoints on a Pi takes minutes; poll rather than fail."""
        for attempt in range(attempts):
            try:
                return self.health()
            except LayaUnavailable:
                if attempt == attempts - 1:
                    raise
                time.sleep(delay)

    def predict(self, state, questions):
        return self._post("/v1/systemone", {
            "state": state,
            "questions": questions,
            "model": self.model,
        })


class LayaPolicy:
    def __init__(self, *, guarded=True, client=None):
        self.client = client or LayaClient()
        self.guarded = guarded
        health = self.client.wait_until_ready()
        # Derive the engine label from whichever server answered: the ONNX service
        # reports {"backend": "onnx"}, stock laya-serve reports {"device": ...}.
        if health.get("backend") == "onnx":
            engine = "ONNX Runtime · fp32"
            name = f"laya-onnx · {self.client.model}"
        else:
            engine = f"laya-serve · torch {health.get('device', 'cpu')}"
            name = f"laya-serve · {self.client.model}"
        self.metadata = {
            "name": name,
            "hardware": platform.machine(),
            "engine": engine,
            "network": "LAN service",
            "server": self.client.base_url,
            "health": health,
            "policy": "Laya probabilities over planner features; optional cycle safety shield",
        }

    def decide(self, game):
        started = time.perf_counter()
        moves = game.moves()
        safe = [m for m in moves if m.safe]
        if not safe and self.guarded:
            raise RuntimeError("Cycle safety invariant violated: no safe action")
        preferred = max(safe, key=lambda m: m.advance).direction if safe else "NONE"
        reachable, space = game.food_reachability()
        state = (
            f"Safe route: {'yes' if safe else 'no'}. "
            f"Food reachable through empty cells: {'yes' if reachable else 'no'}."
        )
        # Only the move choice goes to the model now. The risk/food gauges used to be
        # two extra noul questions (~90 more tokens, ~3x the latency); they are derived
        # here from the deterministic planner instead -- free, exact, and the shield
        # (not the model's noul) is what actually keeps the snake alive.
        questions = {
            "move": {
                "type": "choice",
                "instructions": "Choose the best safe move toward food.",
                "criteria": {
                    m.direction: (
                        "Blocked. Collision."
                        if not m.legal
                        else "Unsafe. Traps the snake."
                        if not m.safe
                        else "Safe. Eat food now. Best."
                        if m.eats
                        else "Safe. Best route to food."
                        if m.direction == preferred
                        else "Safe. Slower route."
                    )
                    for m in moves
                },
            },
        }

        inference_start = time.perf_counter()
        output = self.client.predict(state, questions)
        inference_ms = (time.perf_counter() - inference_start) * 1000

        answers = output["answers"]
        probabilities = _probabilities(answers["move"], DIRECTIONS)
        if any(not math.isfinite(v) or not 0 <= v <= 1 for v in probabilities.values()):
            raise ValueError("Model returned an invalid probability; no move executed")

        # Planner-derived gauges: fraction of free cells still reachable from the head.
        # `space` counts the head too, so subtract it before dividing by the free cells.
        free_cells = game.capacity - len(game.body)
        reach_fraction = min(1.0, max(0.0, (space - 1) / max(1, free_cells)))
        dead_end_risk = 1 - reach_fraction               # high when boxed into a pocket
        food_reachable = reach_fraction if reachable else 0.0

        proposed = max(DIRECTIONS, key=probabilities.__getitem__)
        allowed = [m.direction for m in safe]
        executed = (
            max(allowed, key=probabilities.__getitem__)
            if self.guarded and proposed not in allowed
            else proposed
        )
        usage = output.get("usage", {})
        return Decision(
            probabilities=probabilities,
            proposed=proposed,
            executed=executed,
            safe_directions=allowed,
            intervened=proposed != executed,
            dead_end_risk=dead_end_risk,
            food_reachable=food_reachable,
            inference_ms=inference_ms,
            decision_ms=(time.perf_counter() - started) * 1000,
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            safe_count=len(safe),
            planner_best=preferred,
        )
