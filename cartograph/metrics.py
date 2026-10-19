"""Explicit direct-gain metrics; unjudged results contribute zero gain."""

import math

GAINS = {"E": 1.0, "S": 0.1, "C": 0.01, "I": 0.0}


def evaluate(ranked_ids: list[str], judgments: dict[str, str], k: int = 10) -> dict:
    if k <= 0:
        raise ValueError("k must be positive")
    if len(ranked_ids) != len(set(ranked_ids)):
        raise ValueError("Duplicate ranked IDs would inflate metrics")
    if any(label not in GAINS for label in judgments.values()):
        raise ValueError("Unknown ESCI label")
    top = ranked_ids[:k]
    observed = [GAINS.get(judgments.get(item), 0.0) for item in top]
    ideal = sorted((GAINS[label] for label in judgments.values()), reverse=True)[:k]

    def dcg(gains):
        return sum(gain / math.log2(rank + 2) for rank, gain in enumerate(gains))

    ideal_dcg = dcg(ideal)
    relevant = {key for key, label in judgments.items() if label in {"E", "S"}}
    return {
        "k": k,
        "ndcg": dcg(observed) / ideal_dcg if ideal_dcg else None,
        "judged_positive_recall": len(set(top) & relevant) / len(relevant) if relevant else None,
        "judged_fraction_of_returned": sum(item in judgments for item in top) / len(top) if top else None,
        "returned": len(top),
        "gain_convention": GAINS.copy(),
    }
