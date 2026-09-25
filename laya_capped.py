"""laya-serve with auto-routing kept on but only one checkpoint resident.

Stock laya-serve caps at 2 (Router default); with the ONNX service also running,
holding english+multilingual leaves <400MB free on this Pi (no swap). This caps
`max_loaded=1`: language detection still picks the right checkpoint, but a switch
evicts the previous one instead of stacking it -- trading a one-time reload on
each language switch (~30s) for a safe RAM margin.

Config: LAYA_MODEL (preloaded checkpoint, default multilingual), plus the usual
LAYA_HOST / LAYA_PORT / LAYA_DEVICE / LAYA_THREADS / LAYA_AUTO_TASK.
"""

import os

from laya.router import Router
from laya.serve import create_app, _apply_thread_limit, _resolve_port


def build_app():
    _apply_thread_limit()
    device = os.environ.get("LAYA_DEVICE") or None
    preload = os.environ.get("LAYA_MODEL", "multilingual")
    # max_loaded=1: routing is unchanged, but only one checkpoint is ever resident.
    router = Router(
        device=device,
        max_loaded=1,
        auto_task_detection=os.environ.get("LAYA_AUTO_TASK", "0") not in ("0", "", "false"),
    )
    router.preload([preload])
    return create_app(router)


def main():
    import uvicorn

    uvicorn.run(
        build_app(),
        host=os.environ.get("LAYA_HOST", "0.0.0.0"),
        port=_resolve_port(),
        log_level=os.environ.get("LAYA_LOG_LEVEL", "info"),
    )


if __name__ == "__main__":
    main()
