"""Small BM25 reference implementation for baseline and fixture evaluation."""

import math
import re
from collections import Counter


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.casefold())


class BM25:
    def __init__(self, documents: dict[str, str], k1: float = 1.5, b: float = 0.75):
        if not documents or k1 <= 0 or not 0 <= b <= 1:
            raise ValueError("Need documents, k1 > 0, and 0 <= b <= 1")
        self.k1, self.b = k1, b
        self.counts = {key: Counter(tokenize(value)) for key, value in documents.items()}
        self.lengths = {key: sum(value.values()) for key, value in self.counts.items()}
        self.average_length = sum(self.lengths.values()) / len(documents) or 1.0
        self.frequency = Counter(term for counts in self.counts.values() for term in counts)

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        if k <= 0:
            raise ValueError("k must be positive")
        terms = set(tokenize(query))
        results = []
        for product_id, counts in self.counts.items():
            score = 0.0
            for term in terms:
                frequency = counts[term]
                if not frequency:
                    continue
                df = self.frequency[term]
                idf = math.log(1 + (len(self.counts) - df + 0.5) / (df + 0.5))
                denominator = frequency + self.k1 * (
                    1 - self.b + self.b * self.lengths[product_id] / self.average_length
                )
                score += idf * frequency * (self.k1 + 1) / denominator
            if score > 0:
                results.append((product_id, score))
        return sorted(results, key=lambda item: (-item[1], item[0]))[:k]


class SparseBM25:
    """Same scoring and tie-breaking as BM25, precomputed as a sparse term-weight matrix.

    Needs numpy and scipy; the reference BM25 above stays dependency-free.
    """

    def __init__(self, documents: dict[str, str], k1: float = 1.5, b: float = 0.75):
        import numpy as np
        from scipy import sparse

        if not documents or k1 <= 0 or not 0 <= b <= 1:
            raise ValueError("Need documents, k1 > 0, and 0 <= b <= 1")
        self.k1, self.b = k1, b
        self.ids = list(documents)
        self.vocabulary: dict[str, int] = {}
        rows, cols, tfs = [], [], []
        lengths = np.zeros(len(self.ids))
        for row, key in enumerate(self.ids):
            counts = Counter(tokenize(documents[key]))
            lengths[row] = sum(counts.values())
            for term, frequency in counts.items():
                rows.append(row)
                cols.append(self.vocabulary.setdefault(term, len(self.vocabulary)))
                tfs.append(frequency)
        rows, cols, tfs = np.array(rows), np.array(cols), np.array(tfs, dtype=np.float64)
        n = len(self.ids)
        average_length = lengths.sum() / n or 1.0
        df = np.bincount(cols, minlength=len(self.vocabulary)).astype(np.float64)
        idf = np.log(1 + (n - df + 0.5) / (df + 0.5))
        weights = idf[cols] * tfs * (k1 + 1) / (
            tfs + k1 * (1 - b + b * lengths[rows] / average_length))
        self.weights = sparse.csc_matrix((weights, (rows, cols)), shape=(n, len(self.vocabulary)))
        order = sorted(range(n), key=self.ids.__getitem__)
        self.id_rank = np.empty(n, dtype=np.int64)
        self.id_rank[order] = np.arange(n)

    def scores(self, query: str):
        """BM25 score for every document, aligned with ``self.ids``."""
        import numpy as np

        columns = sorted({self.vocabulary[t] for t in set(tokenize(query)) if t in self.vocabulary})
        if not columns:
            return np.zeros(len(self.ids))
        return np.asarray(self.weights[:, columns].sum(axis=1)).ravel()

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        import numpy as np

        if k <= 0:
            raise ValueError("k must be positive")
        scores = self.scores(query)
        candidates = np.flatnonzero(scores > 0)
        if len(candidates) > k:
            # Keep everything tied with the k-th score so ID tie-breaking matches BM25.
            kth = np.partition(scores[candidates], len(candidates) - k)[len(candidates) - k]
            candidates = candidates[scores[candidates] >= kth]
        order = np.lexsort((self.id_rank[candidates], -scores[candidates]))[:k]
        return [(self.ids[i], float(scores[i])) for i in candidates[order]]
