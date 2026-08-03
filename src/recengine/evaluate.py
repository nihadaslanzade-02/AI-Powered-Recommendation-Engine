"""Top-N ranking metrics and the evaluation loop.

The original measured RMSE. RMSE answers "how close is the predicted number to
the true number", which is the right question for a star-rating system and the
wrong one here: a customer is shown a short list of products and either finds
something worth buying in it or does not. Nothing in that experience depends on
the magnitude of a score, only on the order.

So everything here is computed from a ranked list against a set of products the
customer actually went on to buy. Relevance is binary; there is no graded
relevance to be had from a purchase log.

Two diagnostics sit next to the accuracy metrics on purpose. Catalogue coverage
and novelty are what expose a recommender that scores respectably by pushing
the same few bestsellers at everyone, which is exactly the failure mode a
popularity baseline has and an accuracy number alone will not show.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import Interactions

DEFAULT_KS: tuple[int, ...] = (5, 10, 20)


def precision_at_k(hits: np.ndarray, k: int) -> np.ndarray:
    """Share of the top k that was relevant, per user."""
    return hits[:, :k].sum(axis=1) / k


def recall_at_k(hits: np.ndarray, n_relevant: np.ndarray, k: int) -> np.ndarray:
    """Share of everything relevant that the top k found, per user."""
    return hits[:, :k].sum(axis=1) / n_relevant


def hit_rate_at_k(hits: np.ndarray, k: int) -> np.ndarray:
    """Whether the top k contained anything relevant at all, per user."""
    return (hits[:, :k].sum(axis=1) > 0).astype(float)


def average_precision_at_k(
    hits: np.ndarray, n_relevant: np.ndarray, k: int
) -> np.ndarray:
    """Mean of the precision measured at each position that scored a hit.

    Normalised by ``min(k, n_relevant)`` rather than by ``k``, so a user with
    three relevant products can still reach 1.0 within a list of ten.
    """
    top = hits[:, :k]
    ranks = np.arange(1, k + 1, dtype=float)
    precision_at_each = np.cumsum(top, axis=1) / ranks
    achievable = np.minimum(n_relevant, k)
    return (precision_at_each * top).sum(axis=1) / achievable


def ndcg_at_k(hits: np.ndarray, n_relevant: np.ndarray, k: int) -> np.ndarray:
    """Discounted cumulative gain over the ideal ranking, per user.

    With binary relevance the ideal ranking is simply every relevant product
    first, so the ideal gain depends only on how many the user has.
    """
    top = hits[:, :k]
    discount = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = (top * discount).sum(axis=1)

    ideal_len = np.minimum(n_relevant, k).astype(int)
    cumulative_ideal = np.concatenate([[0.0], np.cumsum(discount)])
    idcg = cumulative_ideal[ideal_len]
    return dcg / idcg


def catalogue_coverage(recommended: np.ndarray, n_items: int, k: int) -> float:
    """Share of the catalogue that appears in anyone's top k."""
    return len(np.unique(recommended[:, :k])) / n_items


def novelty(recommended: np.ndarray, popularity: np.ndarray, n_users: int, k: int) -> float:
    """Mean self-information of the recommended products, in bits.

    ``-log2(share of customers who bought it)``. A recommender that only ever
    proposes the single most popular product scores near zero; one that reaches
    into the tail scores high. Reported beside accuracy because the two trade
    off, and a number that hides the trade-off is not worth reporting.
    """
    share = popularity / n_users
    return float(-np.log2(share[recommended[:, :k]]).mean())


def build_hits(recommended: np.ndarray, relevant: list[np.ndarray]) -> np.ndarray:
    """Boolean (n_users, k): was the product at this rank one the user bought?"""
    hits = np.zeros(recommended.shape, dtype=float)
    for row, wanted in enumerate(relevant):
        hits[row] = np.isin(recommended[row], wanted, assume_unique=True)
    return hits


@dataclass(frozen=True)
class Evaluation:
    """Metrics for one model, one row per cutoff k."""

    model: str
    table: pd.DataFrame
    n_users: int
    fit_seconds: float
    recommend_seconds: float

    def to_frame(self) -> pd.DataFrame:
        out = self.table.copy()
        out.insert(0, "model", self.model)
        out["n_users"] = self.n_users
        out["fit_seconds"] = round(self.fit_seconds, 3)
        out["recommend_seconds"] = round(self.recommend_seconds, 3)
        return out


def score_recommendations(
    recommended: np.ndarray,
    relevant: list[np.ndarray],
    train: Interactions,
    ks: tuple[int, ...] = DEFAULT_KS,
) -> pd.DataFrame:
    """Turn a (n_users, max_k) array of ranked products into a metric table."""
    if recommended.shape[1] < max(ks):
        raise ValueError(
            f"recommendations are {recommended.shape[1]} long but k up to "
            f"{max(ks)} was requested"
        )

    hits = build_hits(recommended, relevant)
    n_relevant = np.array([len(r) for r in relevant], dtype=float)
    popularity = train.item_popularity()

    rows = []
    for k in ks:
        rows.append(
            {
                "k": k,
                "precision": precision_at_k(hits, k).mean(),
                "recall": recall_at_k(hits, n_relevant, k).mean(),
                "map": average_precision_at_k(hits, n_relevant, k).mean(),
                "ndcg": ndcg_at_k(hits, n_relevant, k).mean(),
                "hit_rate": hit_rate_at_k(hits, k).mean(),
                "coverage": catalogue_coverage(recommended, train.n_items, k),
                "novelty_bits": novelty(recommended, popularity, train.n_users, k),
            }
        )
    return pd.DataFrame(rows)


def evaluate_model(
    model,
    train: Interactions,
    relevant: dict[int, np.ndarray],
    ks: tuple[int, ...] = DEFAULT_KS,
    fit_seconds: float = float("nan"),
) -> Evaluation:
    """Rank every evaluable user with ``model`` and score the result.

    ``model`` only has to expose ``recommend(rows, k)``; the exclusion of
    already-seen products lives in the shared base class so that no model can
    gain an advantage by handling it differently.
    """
    import time

    rows = np.array(sorted(relevant), dtype=np.int64)
    wanted = [relevant[int(r)] for r in rows]

    started = time.perf_counter()
    recommended = model.recommend(rows, max(ks))
    elapsed = time.perf_counter() - started

    table = score_recommendations(recommended, wanted, train, ks)
    return Evaluation(
        model=model.name,
        table=table,
        n_users=len(rows),
        fit_seconds=fit_seconds,
        recommend_seconds=elapsed,
    )
