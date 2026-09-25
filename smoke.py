"""Smoke test: confirm the multilingual checkpoint loads and inspect the answer shape."""

import json
import time

import laya

t0 = time.perf_counter()
agent = laya.load("convaiinnovations/laya", subfolder="multilingual")
print(f"load: {time.perf_counter() - t0:.1f}s")

questions = {
    "move": {
        "type": "choice",
        "instructions": "Choose the best safe move toward food.",
        "criteria": {
            "UP": "Blocked. Collision.",
            "DOWN": "Safe. Slower route.",
            "LEFT": "Safe. Best route to food.",
            "RIGHT": "Unsafe. Traps the snake.",
        },
    },
    "risk": {"type": "noul", "instructions": "Is a safe route available?"},
    "food": {"type": "noul", "instructions": "Is food reachable through empty cells?"},
}
state = "Safe route: yes. Food reachable through empty cells: yes."

t0 = time.perf_counter()
out = agent.predict(state, questions)
print(f"predict: {(time.perf_counter() - t0) * 1000:.0f}ms")
print(json.dumps(out, indent=2, ensure_ascii=False, default=str))
