"""RL snake agent: a small PPO-trained CNN run in-process with ONNX Runtime.

No Laya server, no planner. The net sees the board as 6 grids (occupied, steps until
each body cell frees up -- normalised two ways --, food, fill, hunger); only illegal
moves (wall, body, reverse) are masked, which is part of the agent as trained. It
can still trap itself. It is the teacher for the next Laya snake fine-tune.

Adapted from the snake-trap handoff (jev-clones projects/snake-finetune/rl, run ppo1).

Env:
  SNAKE_RL_ONNX     path to the .onnx      (/models/snake-rl/snake-rl.onnx)
  SNAKE_RL_THREADS  intra-op threads       (3; Pi 4 p50 46/26/18/15 ms at 1/2/3/4 threads,
                    but 4 fights laya-onnx for cores and doubles the p95)
Board must be 24x16 (the net's input size).
"""

import math
import os
import platform
import time

import numpy as np

from .game import DIRECTIONS
from .policy import Decision

ONNX_PATH = os.environ.get("SNAKE_RL_ONNX", "/models/snake-rl/snake-rl.onnx")
MODEL_NAME = "snake-rl"
CHANNELS = (
    "occupato (1 = corpo)",
    "passi alla liberazione / celle del tabellone",
    "passi alla liberazione / lunghezza",
    "cibo (1 sulla cella del cibo)",
    "riempimento (lunghezza / celle)",
    "fame (mosse senza cibo / celle, max 2)",
)


def available(path=ONNX_PATH):
    return os.path.isfile(path)


def observe(body, food, hunger, width, height):
    """(1, 6, H, W) float32, identical to rl/snake_env.observe used in training."""
    capacity = width * height
    n = len(body)
    left = np.zeros((height, width), np.float32)
    for i, (x, y) in enumerate(body):
        left[y, x] = n - i
    obs = np.zeros((1, 6, height, width), np.float32)
    obs[0, 0] = left > 0
    obs[0, 1] = left / capacity
    obs[0, 2] = left / n
    if food:
        obs[0, 3, food[1], food[0]] = 1.0
    obs[0, 4] = n / capacity
    obs[0, 5] = min(hunger / capacity, 2.0)
    return obs


class RLPolicy:
    mode = "rl"

    def __init__(self, path=ONNX_PATH):
        import onnxruntime as ort  # Only the RL mode needs it in this process.

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = int(os.environ.get("SNAKE_RL_THREADS", "3"))
        self.session = ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])
        self.hunger, self.last_len, self.last_game = 0, None, None
        self.metadata = {
            "hardware": platform.machine(),
            "engine": "ONNX Runtime · fp32 · in-process",
            "network": "none (local)",
            "server": "-",
            "health": {"status": "ok", "backend": "onnx"},
            "policy": "PPO CNN reads the board grids; illegal moves masked, no planner",
        }

    @property
    def starved(self):
        """Looping without eating: the net's hunger input saturates at 2x the board."""
        return self.last_game is not None and self.hunger > 2 * self.last_game.capacity

    def decide(self, game):
        started = time.perf_counter()
        body, food = list(game.body), game.food
        # New game, or ate on the last move: the hunger counter restarts.
        if game is not self.last_game or self.last_len is None or len(body) > self.last_len:
            self.hunger = 0
        self.last_game, self.last_len = game, len(body)
        obs = observe(body, food, self.hunger, game.width, game.height)

        inference_start = time.perf_counter()
        logits, value = self.session.run(None, {"obs": obs})
        inference_ms = (time.perf_counter() - inference_start) * 1000

        legal = [game.legal_reason(d) == "legal" for d in DIRECTIONS]
        if not any(legal):
            raise RuntimeError("Boxed in: no legal move")  # The runner ends the round as "trapped".
        z = np.where(legal, logits[0].astype(np.float64), -np.inf)
        p = np.exp(z - z.max())
        p /= p.sum()
        probabilities = {d: float(v) for d, v in zip(DIRECTIONS, p)}
        executed = max(DIRECTIONS, key=probabilities.__getitem__)

        safe = [m.direction for m in game.moves() if m.safe]
        reachable, space = game.food_reachability()
        free_cells = game.capacity - len(game.body)
        reach_fraction = min(1.0, max(0.0, (space - 1) / max(1, free_cells)))
        request = self._describe(obs, food, logits[0], float(value[0]), legal)
        self.hunger += 1
        return Decision(
            probabilities=probabilities,
            proposed=executed,
            executed=executed,
            safe_directions=safe,
            intervened=False,
            dead_end_risk=1 - reach_fraction,
            food_reachable=reach_fraction if reachable else 0.0,
            inference_ms=inference_ms,
            decision_ms=(time.perf_counter() - started) * 1000,
            input_tokens=0,
            output_tokens=0,
            safe_count=len(safe),
            planner_best="",
            mode="rl",
            model=MODEL_NAME,
            request=request,
        )

    def _describe(self, obs, food, logits, value, legal):
        """Human-readable view of the tensor in / numbers out, for the page's input panel."""
        _, c, h, w = obs.shape
        grid = obs[0]
        lines = [f"obs: tensore {c} × {h} × {w} (float32), nessun testo, nessun token", ""]
        for i, name in enumerate(CHANNELS):
            if i == 3:
                summary = f"cibo in colonna {food[0] + 1}, riga {food[1] + 1}" if food else "nessun cibo"
            elif i in (4, 5):
                summary = f"{grid[i, 0, 0]:.3f} su tutte le celle"
            else:
                summary = f"{int((grid[i] > 0).sum())} celle ≠ 0, max {grid[i].max():.3f}"
            lines.append(f"canale {i}  {name}\n           {summary}")
        lines += ["", f"fame: {self.hunger} mosse dall'ultimo cibo"]
        return {
            "state": "\n".join(lines),
            "questions": {
                "logits": {d: round(float(v), 3) for d, v in zip(DIRECTIONS, logits)},
                "mascherate (illegali)": [d for d, ok in zip(DIRECTIONS, legal) if not ok],
                "valore stimato": round(value, 3) if math.isfinite(value) else None,
            },
        }
