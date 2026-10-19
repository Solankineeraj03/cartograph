"""Grounded bundle assistant: prompt, structured output parsing, verification, retry, and scoring.

Generators share one interface, ``generate(prompt) -> str``. The deterministic baseline is the
exact solver. LLM generators (vLLM, Transformers) live in ``cartograph.llm``. Raw model behavior is
scored without retries (``raw``); the production wrapper adds JSON extraction, verification, and one
retry with the verifier's errors (``wrapped``). Both are always reported.
"""

import json
import re

from .verifier import Product, verify_bundle

SYSTEM = ("You are Cartograph, a shopping assistant. Use only the catalog evidence provided. "
          "Prices are integer cents and stock is as of the stated time. Reply with one JSON object: "
          '{"decision": "bundle" or "infeasible", "product_ids": [...], "total_cents": int, '
          '"explanation": str}. A bundle has exactly one in-stock product per required category, '
          "no duplicates, and total_cents equal to the sum of the listed prices and within budget. "
          "If no such bundle exists in the evidence, answer infeasible with an empty product_ids list "
          "and total_cents 0, and explain which constraint fails. Cite products only by their IDs.")


def evidence_block(task: dict) -> str:
    lines = [f"As of {task['as_of']} (synthetic prices and stock, feature snapshot {task['feature_export']})."]
    for e in task["evidence"]:
        stock = "in stock" if e["in_stock"] else "OUT OF STOCK"
        lines.append(f"- [{e['product_id']}] {e['category']} | {e['price_cents']} cents | {stock} | {e['title']}")
    return "\n".join(lines)


def user_prompt(task: dict) -> str:
    categories = ", ".join(task["required_categories"])
    return (f"Request: {task['request']}\nRequired categories: {categories}\n"
            f"Budget: {task['budget_cents']} cents\nCatalog evidence:\n{evidence_block(task)}")


def messages(task: dict) -> list[dict]:
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_prompt(task)}]


def parse(text: str) -> tuple[dict | None, str]:
    """Strict: the whole reply must be one JSON object. Lenient: the first {...} span that parses."""
    try:
        value = json.loads(text.strip())
        return (value, "strict") if isinstance(value, dict) else (None, "not_object")
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.S)
    if match:
        try:
            value = json.loads(match.group(0))
            if isinstance(value, dict):
                return value, "lenient"
        except json.JSONDecodeError:
            pass
    return None, "unparseable"


def score(task: dict, reply: dict | None) -> dict:
    """Verifier-grounded score for one reply. Correct abstention requires true infeasibility."""
    feasible = task["solver"]["feasible"]
    evidence = {e["product_id"]: e for e in task["evidence"]}
    catalog = {pid: Product(pid, e["category"], e["price_cents"], e["in_stock"]) for pid, e in evidence.items()}
    out = {"feasible": feasible, "format_ok": False, "decision": None, "valid_bundle": False,
           "correct": False, "errors": [], "uncited_ids": [], "optimality_gap": None}
    if reply is None or reply.get("decision") not in {"bundle", "infeasible"}:
        out["errors"].append("invalid_schema")
        return out
    out["decision"] = reply["decision"]
    ids = reply.get("product_ids")
    out["format_ok"] = isinstance(ids, list) and type(reply.get("total_cents")) is int and \
        isinstance(reply.get("explanation", ""), str)
    if reply["decision"] == "infeasible":
        out["correct"] = not feasible and out["format_ok"]
        if feasible:
            out["errors"].append("false_abstention")
        return out
    check = verify_bundle({"product_ids": ids, "total_cents": reply.get("total_cents")}, catalog,
                          task["budget_cents"], set(task["required_categories"]))
    if isinstance(ids, list):
        out["uncited_ids"] = [i for i in ids if isinstance(i, str) and i not in evidence]
        counts = {}
        for i in ids:
            if i in evidence:
                counts[evidence[i]["category"]] = counts.get(evidence[i]["category"], 0) + 1
        if any(v > 1 for v in counts.values()) or len(ids) != len(task["required_categories"]):
            check["errors"].append("not_one_per_category")
            check["valid"] = False
    out["errors"] += check["errors"]
    out["valid_bundle"] = check["valid"]
    out["correct"] = check["valid"]
    if check["valid"] and feasible:
        chosen = sum(evidence[i]["retrieval_score"] for i in ids)
        out["optimality_gap"] = round(task["solver"]["score"] - chosen, 6)
    if not feasible and check["valid"]:
        out["errors"].append("solver_disagreement")  # verified bundle on a task the exact solver called infeasible
    return out


def feedback(result: dict) -> str:
    return ("Your previous answer failed verification with errors: " + ", ".join(result["errors"]) +
            ". Reply again with one corrected JSON object.")


def run_task(task: dict, generate, retries: int = 1) -> dict:
    """Raw attempt plus wrapped attempts. ``generate(messages) -> str``."""
    history = messages(task)
    text = generate(history)
    reply, mode = parse(text)
    raw = score(task, reply if mode == "strict" else None)
    attempts = [{"text": text, "parse": mode, "score": score(task, reply)}]
    while not attempts[-1]["score"]["correct"] and len(attempts) <= retries:
        history = history + [{"role": "assistant", "content": text},
                             {"role": "user", "content": feedback(attempts[-1]["score"])}]
        text = generate(history)
        reply, mode = parse(text)
        attempts.append({"text": text, "parse": mode, "score": score(task, reply)})
    final = attempts[-1]["score"]
    # Production behavior: never show an unverified bundle; fall back to an explicit refusal.
    shown = "verified_bundle" if final["valid_bundle"] else "abstain_with_reason" if final["decision"] == "infeasible" \
        else "withheld_unverified"
    return {"task_id": task["task_id"], "raw": raw, "wrapped": final, "attempts": attempts, "shown": shown}


def solver_reply(task: dict) -> str:
    solution = task["solver"]
    if solution["feasible"]:
        return json.dumps({"decision": "bundle", **solution["bundle"],
                           "explanation": "Highest-relevance in-stock combination within budget."})
    return json.dumps({"decision": "infeasible", "product_ids": [], "total_cents": 0,
                       "explanation": "No valid bundle: " + ", ".join(solution["reasons"])})
