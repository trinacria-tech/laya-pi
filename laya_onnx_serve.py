"""A minimal laya-serve-compatible HTTP server backed by ONNX Runtime.

Stock `laya-serve` is torch-only. This exposes the same `POST /v1/systemone`
(+ `GET /health`) wire protocol over a single `ONNXAgent`, so the snake (and any
Jev client) can point at it unchanged and get the ORT fp32 speedup.

fp32, not int8: on this Cortex-A72 int8 is faster and smaller but visibly degrades
the yes/no (noul) heads the demo displays (risk/food gauges), while the fp32 export
matches torch numerically.

Several models can be selectable, but only ONE is ever resident: a request naming
a different model (or POST /v1/models/load) drops the current session before the
next one is built, so RAM never holds two (8GB Pi, no swap).

Config via env:
  ONNX_MODELS  JSON registry {name: {"repo" | "path": ..., "file": ...}}; "repo" is an
               HF repo, "path" a local dir (rl_agent_config.json + tokenizer/ + onnx).
               Defaults to {ONNX_NAME: {"repo": ONNX_REPO, "file": ONNX_FILE}}.
  ONNX_NAME    name of the default model    (multilingual)
  ONNX_REPO    HF repo with the export      (soyelmismo/laya-multilingual-onnx)
  ONNX_FILE    onnx file within the repo    (model-fp32.onnx)
  ONNX_DEFAULT model loaded at boot         (first registry entry)
  LAYA_HOST / LAYA_PORT                      (0.0.0.0 / 8001)
  ORT_THREADS  intra-op threads             (cores - 1)
"""

import gc
import json
import os
import threading
import time

REPO = os.environ.get("ONNX_REPO", "soyelmismo/laya-multilingual-onnx")
ONNX_FILE = os.environ.get("ONNX_FILE", "model-fp32.onnx")
NAME = os.environ.get("ONNX_NAME", "multilingual")


def registry():
    raw = os.environ.get("ONNX_MODELS")
    models = json.loads(raw) if raw else {NAME: {"repo": REPO, "file": ONNX_FILE}}
    for name, spec in models.items():
        if not isinstance(spec, dict) or ("repo" in spec) == ("path" in spec):
            raise ValueError(f"ONNX_MODELS[{name!r}] needs exactly one of 'repo' or 'path'")
        spec.setdefault("file", "model-fp32.onnx")
    return models


def load_agent(spec):
    from laya.onnx_agent import ONNXAgent

    if "path" in spec:
        source = spec["path"]
        onnx_path = os.path.join(source, spec["file"])
    else:
        from huggingface_hub import hf_hub_download

        source = spec["repo"]
        onnx_path = hf_hub_download(source, spec["file"])

    agent = ONNXAgent(source, onnx_path=onnx_path)

    # ONNXAgent builds its InferenceSession with no SessionOptions, so intra-op
    # defaults to all 4 cores -- an oversubscription regression when laya-serve
    # shares the box. Rebuild the session with tuned options (per ORT perf docs):
    #   intra_op_num_threads capped (leave a core for laya + snake + uvicorn),
    #   full graph optimization, and SEQUENTIAL execution (a BERT encoder has few
    #   branches, so parallel-op mode only adds scheduling overhead here).
    import onnxruntime as ort

    threads = int(os.environ.get("ORT_THREADS") or max(1, (os.cpu_count() or 2) - 1))
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    agent.session = ort.InferenceSession(
        onnx_path, sess_options=so, providers=["CPUExecutionProvider"]
    )
    return agent


class SingleResident:
    """Holds exactly one loaded model; switching drops the old one before loading."""

    def __init__(self, models, default):
        self.models = models
        self.lock = threading.Lock()  # Serialises swaps and inference on the one session.
        self.name = self.agent = None
        self.load_s = 0.0
        self.ensure(default)

    def ensure(self, name):
        if name not in self.models:
            raise KeyError(name)
        with self.lock:
            if name == self.name:
                return
            # Free the resident session first: two fp32 encoders don't fit alongside
            # the rest of the stack.
            self.name = self.agent = None
            gc.collect()
            started = time.perf_counter()
            self.agent = load_agent(self.models[name])
            self.name, self.load_s = name, time.perf_counter() - started

    def predict(self, name, state, questions):
        if name is not None:
            self.ensure(name)
        with self.lock:
            out = self.agent.predict(state, questions)
            return {**out, "model": self.name}

    def describe(self):
        return {
            "loaded": self.name,
            "load_seconds": round(self.load_s, 2),
            "models": [
                {"name": n, "source": s.get("repo") or s.get("path"), "file": s["file"]}
                for n, s in self.models.items()
            ],
        }


def build_app():
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.concurrency import run_in_threadpool

    models = registry()
    resident = SingleResident(models, os.environ.get("ONNX_DEFAULT") or next(iter(models)))

    app = FastAPI(title="laya-onnx", summary="Laya System-1 decisions via ONNX Runtime")

    @app.get("/health")
    def health():
        spec = models[resident.name] if resident.name else {}
        return {"status": "ok", "backend": "onnx", "model": resident.name,
                "repo": spec.get("repo") or spec.get("path"), "file": spec.get("file")}

    @app.get("/v1/models")
    def list_models():
        return resident.describe()

    @app.post("/v1/models/load")
    async def load_model(request: Request):
        try:
            name = (await request.json())["model"]
        except Exception:
            raise HTTPException(status_code=400, detail="body must be {\"model\": <name>}")
        try:
            await run_in_threadpool(resident.ensure, name)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"unknown model {name!r}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"load failed: {e}")
        return resident.describe()

    @app.post("/v1/systemone")
    async def systemone(request: Request):
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="request body must be valid JSON")
        if not isinstance(body, dict) or "questions" not in body:
            raise HTTPException(status_code=400, detail="body must be an object with 'questions'")
        if body.get("model") is not None and body["model"] not in models:
            raise HTTPException(status_code=404, detail=f"unknown model {body['model']!r}")
        try:
            # Offload the ~2.5s synchronous ORT call so /health and the event loop stay live.
            # No "model" = whatever is resident (the showcase relies on this); a named
            # model swaps in first if it isn't the resident one.
            out = await run_in_threadpool(
                resident.predict, body.get("model"), body.get("state"), body["questions"]
            )
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        except Exception:
            raise HTTPException(status_code=500, detail="inference failed")
        # {model, answers, usage}; model is the registry name that actually answered.
        return out

    return app


def main():
    import uvicorn

    uvicorn.run(
        build_app(),
        host=os.environ.get("LAYA_HOST", "0.0.0.0"),
        port=int(os.environ.get("LAYA_PORT", "8001")),
        log_level=os.environ.get("LAYA_LOG_LEVEL", "info"),
    )


if __name__ == "__main__":
    main()
