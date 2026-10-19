"""Ray Serve application: the served Cartograph request path plus an OpenAI-style completions endpoint.

POST /v1/bundle  {"request": str, "now": ISO time (simulated clock, optional)}
    validate -> per-category retrieval (BM25 over titles + category) -> live features from Feast/Redis
    (freshness from the materialization watermark) -> exact solver (feasibility oracle) -> LLM answer
    (vLLM engine in-process, greedy) -> verify, one retry -> verified bundle, refusal with reason, or withheld.
    Every response carries a trace: release, model, feature watermark, evidence IDs, verifier result, stage timings.
POST /v1/completions  streaming SSE in the OpenAI completions format, so `vllm bench serve` can measure this
    deployment with the same prompts as a bare `vllm serve`.
GET /health
"""

import json
import os
import time
import uuid
from pathlib import Path

from ray import serve
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse

from .assistant import messages, parse, score, feedback
from .service import InvalidRequest, parse_request
from .solver import Candidate, solve
from .verifier import Product

CONFIG = {
    "model": os.environ.get("CARTOGRAPH_MODEL", ""),
    "release_id": os.environ.get("CARTOGRAPH_RELEASE", "unreleased"),
    "slice": os.environ.get("CARTOGRAPH_SLICE", ""),
    "sim": os.environ.get("CARTOGRAPH_SIM", ""),
    "export": os.environ.get("CARTOGRAPH_EXPORT", ""),
    "feature_repo": os.environ.get("CARTOGRAPH_FEATURE_REPO", ""),
    "redis_port": int(os.environ.get("CARTOGRAPH_REDIS_PORT", "6390")),
    # The demo page calls this server from a browser through an SSH tunnel; set to the page's origin to restrict.
    "cors_origin": os.environ.get("CARTOGRAPH_CORS_ORIGIN", "*"),
}
CORS = {"Access-Control-Allow-Origin": CONFIG["cors_origin"], "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type", "Access-Control-Max-Age": "600"}


@serve.deployment(ray_actor_options={"num_gpus": 1}, max_ongoing_requests=256)
class Generator:
    def __init__(self, model: str):
        from transformers import AutoTokenizer
        from vllm import AsyncEngineArgs, AsyncLLMEngine
        self.engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(
            model=model, max_model_len=2048, gpu_memory_utilization=0.80, enable_prefix_caching=True, seed=0))
        self.tokenizer = AutoTokenizer.from_pretrained(model)

    async def chat(self, history: list[dict], max_tokens: int = 256) -> str:
        from vllm import SamplingParams
        prompt = self.tokenizer.apply_chat_template(history, add_generation_prompt=True, tokenize=False)
        final = None
        async for out in self.engine.generate(prompt, SamplingParams(temperature=0.0, max_tokens=max_tokens),
                                              str(uuid.uuid4())):
            final = out
        return final.outputs[0].text

    async def stream(self, prompt: str, params: dict):
        from vllm import SamplingParams
        sampling = SamplingParams(temperature=params.get("temperature", 0.0), max_tokens=params.get("max_tokens", 16),
                                  min_tokens=params.get("min_tokens", 0), ignore_eos=params.get("ignore_eos", False))
        sent, final = 0, None
        async for out in self.engine.generate(prompt, sampling, str(uuid.uuid4())):
            text = out.outputs[0].text
            if len(text) > sent:
                yield {"text": text[sent:]}
                sent = len(text)
            final = out
        yield {"usage": {"prompt_tokens": len(final.prompt_token_ids),
                         "completion_tokens": len(final.outputs[0].token_ids)}}


@serve.deployment(max_ongoing_requests=256)
class Cartograph:
    """Routes in __call__ rather than a FastAPI ingress: newer FastAPI app objects hold a thread lock that Ray 2.49
    can't serialize (the first serve run failed with 'Failed to serialize the ASGI app')."""
    def __init__(self, generator):
        from feast import FeatureStore

        from .online import FeatureClient
        from .tasks import Catalog
        self.generator = generator
        self.catalog = Catalog(Path(CONFIG["slice"]), Path(CONFIG["sim"]), Path(CONFIG["export"]))
        self.features = FeatureClient(FeatureStore(repo_path=CONFIG["feature_repo"]), CONFIG["redis_port"])

    async def __call__(self, request: Request):
        route = (request.method, request.url.path)
        if request.method == "OPTIONS":  # CORS preflight from the demo page
            from starlette.responses import Response
            return Response(status_code=204, headers=CORS)
        if route == ("GET", "/health"):
            response = await self.health()
        elif route == ("GET", "/v1/info"):
            response = await self.info()
        elif route == ("POST", "/v1/completions"):
            response = await self.completions(request)
        elif route == ("POST", "/v1/bundle"):
            response = await self.bundle(request)
        else:
            response = JSONResponse({"error": f"no route {route}"}, status_code=404)
        response.headers.update(CORS)
        return response

    async def info(self):
        import json as _json

        import redis
        watermark = redis.Redis(host="127.0.0.1", port=CONFIG["redis_port"], socket_timeout=1).get(
            "cartograph:feature_watermark")
        return JSONResponse({"release": CONFIG["release_id"], "model": Path(CONFIG["model"]).name,
                             "feature_watermark": _json.loads(watermark) if watermark else None,
                             "catalog_products": int(len(self.catalog.products)),
                             "categories": sorted(self.catalog.products.category.unique().tolist())})

    async def health(self):
        return JSONResponse({"ok": True, "release": CONFIG["release_id"]})

    async def completions(self, request: Request):
        body = await request.json()
        handle = self.generator.options(stream=True)

        async def events():
            async for item in handle.stream.remote(body["prompt"], body):
                if "text" in item:
                    yield "data: " + json.dumps({"choices": [{"index": 0, "text": item["text"]}]}) + "\n\n"
                else:
                    yield "data: " + json.dumps({"choices": [], "usage": item["usage"]}) + "\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(events(), media_type="text/event-stream")

    async def bundle(self, request: Request):
        from datetime import datetime, timezone
        body = await request.json()
        now = datetime.fromisoformat(body["now"]) if body.get("now") else datetime.now(timezone.utc)
        t0 = time.perf_counter()
        trace = {"request_id": str(uuid.uuid4()), "release_id": CONFIG["release_id"], "model": CONFIG["model"],
                 "request": body.get("request", ""), "now": now.isoformat(), "timings_ms": {}}
        try:
            parsed = parse_request(trace["request"])
        except InvalidRequest as error:
            trace.update(decision="invalid_request", error=str(error))
            return JSONResponse(trace)
        trace["parsed"] = parsed
        t1 = time.perf_counter()
        hits = []
        for category in parsed["required_categories"]:
            query = f"{parsed['adjective']} {category.replace('_', ' ')}".strip()
            ranked = [(pid, s) for pid, s in self.catalog.indexes["test"].search(query, k=200)
                      if self.catalog.products.loc[pid, "category"] == category][:8]
            hits += [(pid, s, category) for pid, s in ranked]
        t2 = time.perf_counter()
        from .online import FeaturesUnavailable
        try:
            fetched = self.features.fetch([pid for pid, _, _ in hits], now=now)
        except FeaturesUnavailable as error:
            trace.update(decision="features_unavailable", error=str(error), timings_ms={"total": (time.perf_counter() - t0) * 1000})
            return JSONResponse(trace, status_code=503)
        t3 = time.perf_counter()
        trace["feature_watermark"] = fetched["watermark"]
        trace["feature_snapshot_age_s"] = fetched["snapshot_age_s"]
        if fetched["stale"]:
            trace.update(decision="stale_features", reason="feature snapshot older than the freshness limit; "
                         "no availability or price claims made", evidence_ids=[pid for pid, _, _ in hits],
                         evidence=[{"product_id": pid, "category": category, "retrieval_score": round(float(s), 4),
                                    "title": str(self.catalog.products.loc[pid, "product_title"])[:120]}
                                   for pid, s, category in hits])
            trace["timings_ms"] = {"validate": (t1 - t0) * 1000, "retrieve": (t2 - t1) * 1000, "features": (t3 - t2) * 1000}
            return JSONResponse(trace)
        evidence = []
        for pid, s, category in hits:
            f = fetched["products"][pid]
            if f["status"] != "ok":
                continue
            evidence.append({"product_id": pid, "category": category, "price_cents": int(f["price_cents"]),
                             "in_stock": bool(f["in_stock"]), "retrieval_score": round(float(s), 4),
                             "title": str(self.catalog.products.loc[pid, "product_title"])[:120]})
        pool = [Candidate(Product(e["product_id"], e["category"], e["price_cents"], e["in_stock"]), e["retrieval_score"])
                for e in evidence]
        oracle = solve(pool, parsed["required_categories"], parsed["budget_cents"]) if pool else \
            {"feasible": False, "reasons": ["no_evidence"], "bundle": None}
        task = {"task_id": trace["request_id"], "request": trace["request"], "required_categories": parsed["required_categories"],
                "budget_cents": parsed["budget_cents"], "as_of": now.isoformat(), "evidence": evidence, "solver": oracle,
                "feature_export": (fetched["watermark"] or {}).get("export_tag")}
        t4 = time.perf_counter()
        history = messages(task)
        attempts = []
        for attempt in range(2):
            text = await self.generator.chat.remote(history)
            reply, mode = parse(text)
            result = score(task, reply)
            attempts.append({"text": text, "parse": mode, "score": result})
            if result["correct"]:
                break
            history = history + [{"role": "assistant", "content": text}, {"role": "user", "content": feedback(result)}]
        t5 = time.perf_counter()
        final = attempts[-1]["score"]
        trace.update(evidence_ids=[e["product_id"] for e in evidence], evidence=evidence, oracle=oracle, attempts=attempts,
                     decision="verified_bundle" if final["valid_bundle"] else
                     "abstain_with_reason" if final["decision"] == "infeasible" else "withheld_unverified",
                     answer=parse(attempts[-1]["text"])[0] if final["format_ok"] else None,
                     correct_vs_oracle=final["correct"])
        trace["timings_ms"] = {"validate": (t1 - t0) * 1000, "retrieve": (t2 - t1) * 1000, "features": (t3 - t2) * 1000,
                               "solve": (t4 - t3) * 1000, "generate_and_verify": (t5 - t4) * 1000,
                               "total": (time.perf_counter() - t0) * 1000, "llm_calls": len(attempts)}
        return JSONResponse(trace)


def build():
    return Cartograph.bind(Generator.bind(CONFIG["model"]))


def main():
    import ray
    ray.init(include_dashboard=False)
    serve.start(http_options={"host": "127.0.0.1", "port": int(os.environ.get("CARTOGRAPH_PORT", "8012"))})
    serve.run(build(), name="cartograph", route_prefix="/", blocking=True)


if __name__ == "__main__":
    main()
