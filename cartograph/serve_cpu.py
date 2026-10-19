"""CPU-only Ray Serve app for Kubernetes: the deterministic Cartograph request path with a full trace.

Retrieval (BM25 over titles + derived category), as-of features from the bundled export snapshot, exact solver,
verifier. No LLM, so it runs on CPU nodes. The release ID and data bundle path come from the environment, so the
RayService spec pins them and a rollback is a spec change.
"""

import os
import socket
from pathlib import Path

from ray import serve
from starlette.requests import Request
from starlette.responses import JSONResponse

from .release import trace_digest
from .service import answer

@serve.deployment(num_replicas=1, ray_actor_options={"num_cpus": 0.5})
class CartographCPU:
    def __init__(self):
        from .tasks import Catalog
        root = Path(os.environ.get("CARTOGRAPH_BUNDLE", "/app/demo_bundle"))
        self.release = os.environ.get("CARTOGRAPH_RELEASE", "unreleased")
        self.catalog = Catalog(root / "slice", root / "sim", root / "export")

    async def __call__(self, request: Request):
        if request.method == "GET" and request.url.path == "/health":
            return JSONResponse({"ok": True, "release": self.release, "pod": socket.gethostname()})
        if not (request.method == "POST" and request.url.path == "/v1/bundle"):
            return JSONResponse({"error": "not found"}, status_code=404)
        body = await request.json()
        trace = answer(body["request"], self.catalog, body.get("as_of", "2026-09-20T12:00:00+00:00"), self.release)
        trace["digest"] = trace_digest(trace) if trace.get("decision") != "invalid_request" else None
        trace["served_by_pod"] = socket.gethostname()
        trace.pop("evidence", None)
        return JSONResponse(trace)


app = CartographCPU.bind()
