"""De-risk the ONNX path: load a community multilingual export, check output shape
and latency vs the torch number (~4.2s/move)."""

import json
import time

from huggingface_hub import hf_hub_download
from laya.onnx_agent import ONNXAgent

REPO = "soyelmismo/laya-multilingual-onnx"
ONNX_FILE = "model-fp32.onnx"

print(f"downloading {ONNX_FILE} from {REPO} ...")
onnx_path = hf_hub_download(REPO, ONNX_FILE)
print("onnx at", onnx_path)

t0 = time.perf_counter()
agent = ONNXAgent(REPO, onnx_path=onnx_path)
print(f"load: {time.perf_counter()-t0:.1f}s")

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

# warmup
agent.predict(state, questions)

samples = []
for _ in range(8):
    t0 = time.perf_counter()
    out = agent.predict(state, questions)
    samples.append((time.perf_counter() - t0) * 1000)

samples.sort()
print(f"\nlatency ms: median={samples[len(samples)//2]:.0f} min={samples[0]:.0f} max={samples[-1]:.0f}")
print("output keys:", list(out.keys()))
print(json.dumps(out.get("answers", {}), indent=2, default=str)[:600])
print("usage:", out.get("usage"))
