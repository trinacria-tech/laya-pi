"""laya-serve pinned to a single checkpoint.

Stock `laya-serve` builds a Router that auto-routes by language and lazily loads
whichever checkpoint a request implies -- so with port 8000 on the LAN, a client
POSTing English text (or naming another model) can load a *second* checkpoint and,
once three are resident, evict the one the demo depends on. This entrypoint forces
every request onto one checkpoint, so exactly one model is ever built and it is
never evicted.

No CORS is added: browsers only ever talk to the snake page's own origin (:8080);
:8000 is for direct API callers (curl, other services), where same-origin does
not apply.

Config: LAYA_MODEL (default multilingual), LAYA_HOST, LAYA_PORT, LAYA_DEVICE,
LAYA_THREADS, LAYA_API_KEY -- same names laya-serve already honours.
"""

import os

from laya.router import Router, normalise_name
from laya.serve import create_app, _apply_thread_limit, _resolve_port


class SingleModelRouter(Router):
    """A Router that answers every request from one fixed checkpoint.

    Overriding `route` (not `_route`) means the pin also covers the `model=` field
    and language detection uniformly, and `predict`'s `self.load(decision["model"])`
    can only ever ask for the fixed name.
    """

    def __init__(self, fixed: str, **kwargs):
        self.fixed = normalise_name(fixed)
        super().__init__(**kwargs)

    def route(self, state, questions=None, model=None, **kwargs):
        # Force the checkpoint regardless of what the caller asked for.
        return super().route(state, questions, model=self.fixed,
                             task=None, lang=None, lang_guess=None)


def build_app():
    _apply_thread_limit()  # honours LAYA_THREADS before torch spins up its pools
    fixed = os.environ.get("LAYA_MODEL", "multilingual")
    device = os.environ.get("LAYA_DEVICE") or None
    # max_loaded=1: with only one checkpoint reachable this is belt-and-suspenders,
    # but it makes the "never evicted" guarantee structural rather than incidental.
    router = SingleModelRouter(fixed, device=device, max_loaded=1, default=fixed)
    router.preload([fixed])
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
