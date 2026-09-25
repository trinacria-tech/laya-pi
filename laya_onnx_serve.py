"""A minimal laya-serve-compatible HTTP server backed by ONNX Runtime.

Stock `laya-serve` is torch-only. This exposes the same `POST /v1/systemone`
(+ `GET /health`) wire protocol over a single `ONNXAgent`, so the snake (and any
Jev client) can point at it unchanged and get the ORT fp32 speedup.

fp32, not int8: on this Cortex-A72 int8 is faster and smaller but visibly degrades
the yes/no (noul) heads the demo displays (risk/food gauges), while the fp32 export
matches torch numerically.

Config via env:
  ONNX_REPO    HF repo with the export      (soyelmismo/laya-multilingual-onnx)
  ONNX_FILE    onnx file within the repo    (model-fp32.onnx)
  LAYA_HOST / LAYA_PORT                      (0.0.0.0 / 8001)
  ORT_THREADS  intra-op threads             (cores - 1)
"""

import os

REPO = os.environ.get("ONNX_REPO", "soyelmismo/laya-multilingual-onnx")
ONNX_FILE = os.environ.get("ONNX_FILE", "model-fp32.onnx")


def build_app():
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.concurrency import run_in_threadpool
    from huggingface_hub import hf_hub_download
    from laya.onnx_agent import ONNXAgent

    onnx_path = hf_hub_download(REPO, ONNX_FILE)

    agent = ONNXAgent(REPO, onnx_path=onnx_path)

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

    app = FastAPI(title="laya-onnx", summary="Laya System-1 decisions via ONNX Runtime")

    @app.get("/health")
    def health():
        return {"status": "ok", "backend": "onnx", "repo": REPO, "file": ONNX_FILE}

    @app.post("/v1/systemone")
    async def systemone(request: Request):
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="request body must be valid JSON")
        if not isinstance(body, dict) or "questions" not in body:
            raise HTTPException(status_code=400, detail="body must be an object with 'questions'")
        try:
            # Offload the ~2.5s synchronous ORT call so /health and the event loop stay live.
            out = await run_in_threadpool(agent.predict, body.get("state"), body["questions"])
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        except Exception:
            raise HTTPException(status_code=500, detail="inference failed")
        # ONNXAgent already returns {model, answers, usage}; the snake reads answers+usage.
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
