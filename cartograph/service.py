"""Deterministic request path with a full trace: validate -> retrieve -> features -> solve -> verify.

This is the non-LLM production path (and the fallback when the LLM is unavailable). The trace binds
the answer to the release, retrieval index, feature snapshot, evidence IDs, verifier result, and timings.
"""

import re
import time
import uuid

import pandas as pd

from .solver import Candidate, solve
from .verifier import Product, verify_bundle

# Longest phrases first so "keyboard and mouse combo" is not split into keyboard + mouse.
CATEGORY_WORDS = {"keyboard and mouse combo": "keyboard_mouse_combo", "keyboard mouse combo": "keyboard_mouse_combo",
                  "keyboard & mouse combo": "keyboard_mouse_combo", "mouse pad": "mouse_pad",
                  "laptop stand": "laptop_stand", "usb hub": "usb_hub", "keyboard": "keyboard", "mouse": "mouse",
                  "headphones": "headphones", "headset": "headphones", "monitor": "monitor", "webcam": "webcam",
                  "microphone": "microphone", "speaker": "speaker", "charger": "charger"}
ADJECTIVE_WORDS = ["quiet", "silent", "wireless", "ergonomic", "compact", "gaming", "bluetooth", "rgb", "usb"]


class InvalidRequest(ValueError):
    pass


def parse_request(text: str) -> dict:
    budget = re.search(r"\$\s*([0-9]+(?:\.[0-9]{1,2})?)", text)
    if not budget:
        raise InvalidRequest("no budget found (expected e.g. 'under $100')")
    lowered = text.casefold()
    # Categories are returned in the order they appear in the request (the served prompt lists them in that order).
    found, rest = [], lowered
    for phrase in sorted(CATEGORY_WORDS, key=len, reverse=True):
        for match in re.finditer(rf"\b{re.escape(phrase)}\b", rest):
            found.append((match.start(), CATEGORY_WORDS[phrase]))
        rest = re.sub(rf"\b{re.escape(phrase)}\b", lambda m: " " * len(m.group(0)), rest)
    categories = []
    for _, category in sorted(found):
        if category not in categories:
            categories.append(category)
    if not categories:
        raise InvalidRequest("no supported product category found")
    adjectives = [a for a in ADJECTIVE_WORDS if re.search(rf"\b{a}\b", lowered)]
    return {"required_categories": categories, "budget_cents": int(round(float(budget.group(1)) * 100)),
            "adjective": " ".join(adjectives), "require_in_stock": True}


def answer(text: str, catalog, as_of: str, release_id: str, split: str = "test") -> dict:
    trace = {"request_id": str(uuid.uuid4()), "release_id": release_id, "request": text, "as_of": as_of,
             "timings_ms": {}, "feature_snapshot": catalog.features.manifest.get("tag")}
    start = time.perf_counter()
    try:
        request = parse_request(text)
    except InvalidRequest as error:
        trace.update(decision="invalid_request", error=str(error), evidence_ids=[], verifier=None)
        return trace
    trace["parsed"] = request
    t0 = time.perf_counter()
    trace["timings_ms"]["validate"] = (t0 - start) * 1000
    evidence = []
    for category in request["required_categories"]:
        evidence += catalog.candidates(split, category, request["adjective"], pd.Timestamp(as_of))
    t1 = time.perf_counter()
    trace["timings_ms"]["retrieve_and_features"] = (t1 - t0) * 1000
    trace["evidence_ids"] = [e["product_id"] for e in evidence]
    trace["evidence"] = evidence
    pool = [Candidate(Product(e["product_id"], e["category"], e["price_cents"], e["in_stock"]), e["retrieval_score"])
            for e in evidence]
    solution = solve(pool, request["required_categories"], request["budget_cents"])
    t2 = time.perf_counter()
    trace["timings_ms"]["solve"] = (t2 - t1) * 1000
    if solution["feasible"]:
        catalog_map = {c.product.product_id: c.product for c in pool}
        trace["verifier"] = verify_bundle(solution["bundle"], catalog_map, request["budget_cents"],
                                          set(request["required_categories"]))
        trace["decision"] = "bundle" if trace["verifier"]["valid"] else "withheld_unverified"
        trace["bundle"] = solution["bundle"]
    else:
        trace["verifier"] = None
        trace["decision"] = "infeasible"
        trace["reasons"] = solution["reasons"]
    trace["timings_ms"]["total"] = (time.perf_counter() - start) * 1000
    return trace
