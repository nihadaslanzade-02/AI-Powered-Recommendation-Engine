"""Ranking metrics, checked against arithmetic worked out by hand."""

from __future__ import annotations

import numpy as np
import pytest

from recengine.data import build_interactions, load_transactions, temporal_split
from recengine.evaluate import (
    average_precision_at_k,
    build_hits,
    catalogue_coverage,
    hit_rate_at_k,
    ndcg_at_k,
    novelty,
    precision_at_k,
    recall_at_k,
    score_recommendations,
)

from .conftest import CUTOFF

# One user, five recommendations, hits at ranks 2 and 4, three relevant in total.
HITS = np.array([[0.0, 1.0, 0.0, 1.0, 0.0]])
N_RELEVANT = np.array([3.0])


def test_precision_and_recall():
    assert precision_at_k(HITS, 5)[0] == pytest.approx(2 / 5)
    assert precision_at_k(HITS, 3)[0] == pytest.approx(1 / 3)
    assert recall_at_k(HITS, N_RELEVANT, 5)[0] == pytest.approx(2 / 3)
    assert recall_at_k(HITS, N_RELEVANT, 3)[0] == pytest.approx(1 / 3)


def test_hit_rate_is_all_or_nothing():
    assert hit_rate_at_k(HITS, 5)[0] == 1.0
    assert hit_rate_at_k(HITS, 1)[0] == 0.0


def test_average_precision_matches_hand_calculation():
    # Precision is 1/2 at rank 2 and 2/4 at rank 4; three relevant, so the
    # normaliser is min(3, 5) = 3.
    assert average_precision_at_k(HITS, N_RELEVANT, 5)[0] == pytest.approx(
        (0.5 + 0.5) / 3
    )
    assert average_precision_at_k(HITS, N_RELEVANT, 3)[0] == pytest.approx(0.5 / 3)


def test_ndcg_matches_hand_calculation():
    dcg = 1 / np.log2(3) + 1 / np.log2(5)
    idcg = 1 / np.log2(2) + 1 / np.log2(3) + 1 / np.log2(4)
    assert ndcg_at_k(HITS, N_RELEVANT, 5)[0] == pytest.approx(dcg / idcg)


def test_a_perfect_ranking_scores_one():
    hits = np.zeros((1, 5))
    hits[0, :3] = 1.0
    assert average_precision_at_k(hits, N_RELEVANT, 5)[0] == pytest.approx(1.0)
    assert ndcg_at_k(hits, N_RELEVANT, 5)[0] == pytest.approx(1.0)


def test_a_perfect_ranking_scores_one_even_when_relevance_exceeds_k():
    """Normalising by k rather than min(k, n_relevant) would break this."""
    hits = np.ones((1, 5))
    many = np.array([10.0])
    assert average_precision_at_k(hits, many, 5)[0] == pytest.approx(1.0)
    assert ndcg_at_k(hits, many, 5)[0] == pytest.approx(1.0)
    assert recall_at_k(hits, many, 5)[0] == pytest.approx(0.5)


def test_no_hits_scores_zero():
    empty = np.zeros((1, 5))
    assert average_precision_at_k(empty, N_RELEVANT, 5)[0] == 0.0
    assert ndcg_at_k(empty, N_RELEVANT, 5)[0] == 0.0
    assert hit_rate_at_k(empty, 5)[0] == 0.0


def test_build_hits_marks_the_right_positions():
    recommended = np.array([[7, 3, 9], [1, 2, 3]])
    relevant = [np.array([3]), np.array([1, 3])]
    assert build_hits(recommended, relevant).tolist() == [
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
    ]


def test_coverage_counts_distinct_products():
    recommended = np.array([[0, 1], [0, 1], [0, 2]])
    assert catalogue_coverage(recommended, n_items=10, k=2) == pytest.approx(0.3)
    assert catalogue_coverage(recommended, n_items=10, k=1) == pytest.approx(0.1)


def test_novelty_is_higher_for_rarer_products():
    popularity = np.array([50.0, 1.0])
    common = novelty(np.array([[0]]), popularity, n_users=100, k=1)
    rare = novelty(np.array([[1]]), popularity, n_users=100, k=1)
    assert common == pytest.approx(1.0)  # -log2(0.5)
    assert rare > common


def test_score_recommendations_rejects_a_list_shorter_than_k(toy_csv):
    frame = load_transactions(toy_csv)
    train_frame, _ = temporal_split(frame, CUTOFF)
    train = build_interactions(train_frame)
    with pytest.raises(ValueError, match="but k up to"):
        score_recommendations(
            np.zeros((1, 3), dtype=int), [np.array([0])], train, ks=(5,)
        )
