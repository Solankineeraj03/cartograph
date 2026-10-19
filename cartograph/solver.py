"""Exact bundle solver over a bounded candidate set: the non-LLM baseline and feasibility oracle.

Chooses one distinct, in-stock product per required category with total <= budget. The objective
maximizes the summed relevance score; ties go to the lower total, then the lexicographically
smaller ID tuple. If nothing is feasible, it returns the reasons, so abstention can be checked
against ground truth instead of rewarded blindly.
"""

from dataclasses import dataclass
from itertools import product as cartesian

from .verifier import Product, verify_bundle


@dataclass(frozen=True)
class Candidate:
    product: Product
    score: float


def solve(candidates: list[Candidate], required_categories: list[str], budget_cents: int,
          per_category: int = 25) -> dict:
    if len(set(required_categories)) != len(required_categories) or not required_categories:
        raise ValueError("required categories must be non-empty and distinct")
    pools, reasons = [], []
    for category in required_categories:
        in_category = [c for c in candidates if c.product.category == category]
        available = [c for c in in_category if c.product.in_stock]
        if not in_category:
            reasons.append(f"no_candidate:{category}")
        elif not available:
            reasons.append(f"none_in_stock:{category}")
        pools.append(sorted(available, key=lambda c: (-c.score, c.product.price_cents,
                                                      c.product.product_id))[:per_category])
    if reasons:
        return {"feasible": False, "reasons": reasons, "bundle": None}
    best, best_key, cheapest = None, None, None
    for combo in cartesian(*pools):
        ids = [c.product.product_id for c in combo]
        if len(set(ids)) != len(ids):
            continue
        total = sum(c.product.price_cents for c in combo)
        cheapest = total if cheapest is None else min(cheapest, total)
        if total > budget_cents:
            continue
        key = (-sum(c.score for c in combo), total, tuple(ids))
        if best_key is None or key < best_key:
            best, best_key = combo, key
    if best is None:
        return {"feasible": False, "reasons": [f"over_budget:cheapest_{cheapest}_cents"], "bundle": None,
                "cheapest_total_cents": cheapest}
    output = {"product_ids": [c.product.product_id for c in best],
              "total_cents": sum(c.product.price_cents for c in best)}
    catalog = {c.product.product_id: c.product for c in candidates}
    check = verify_bundle(output, catalog, budget_cents, set(required_categories))
    if not check["valid"]:  # the solver must never emit something the verifier rejects
        raise AssertionError(f"solver produced invalid bundle: {check}")
    return {"feasible": True, "reasons": [], "bundle": output, "score": -best_key[0]}
