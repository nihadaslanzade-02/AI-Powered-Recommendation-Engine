"""The recommenders, and the shared behaviour they all depend on."""

from __future__ import annotations

import numpy as np
import pytest

from recengine.data import (
    build_interactions,
    heldout_items,
    load_transactions,
    temporal_split,
)
from recengine.evaluate import build_hits
from recengine.models import (
    ItemKNNRecommender,
    PopularityRecommender,
    PureSVDRecommender,
    Recommender,
    default_models,
)

from .conftest import CUTOFF


@pytest.fixture
def split(toy_csv):
    frame = load_transactions(toy_csv)
    train_frame, test_frame = temporal_split(frame, CUTOFF)
    train = build_interactions(train_frame)
    relevant = heldout_items(test_frame, train)
    return train_frame, train, relevant


class ConstantRecommender(Recommender):
    """Every product scores the same, which is what the original ended up doing."""

    name = "Constant"

    def _fit(self, train, transactions):
        pass

    def score(self, rows):
        return np.ones((len(rows), self.train.n_items))


class NaNRecommender(Recommender):
    name = "NaN"

    def _fit(self, train, transactions):
        pass

    def score(self, rows):
        return np.full((len(rows), self.train.n_items), np.nan)


def test_recommend_never_returns_a_product_the_customer_already_bought(split):
    _, train, _ = split
    model = PureSVDRecommender(n_factors=4).fit(train)
    rows = np.arange(train.n_users)
    recommended = model.recommend(rows, 5)
    for i, row in enumerate(rows):
        assert not np.intersect1d(recommended[i], train.seen_items(row)).size


def test_ties_break_deterministically_by_index(split):
    """The original's real failure mode, pinned.

    Identical scores plus a stable sort meant the "top 10" was the first ten
    candidates in column order. That is still what happens, but now it is a
    stated rule rather than an accident, and it is reproducible.
    """
    _, train, _ = split
    model = ConstantRecommender().fit(train)
    rows = np.array([0])

    first = model.recommend(rows, 5)
    second = model.recommend(rows, 5)
    np.testing.assert_array_equal(first, second)

    unseen = np.setdiff1d(np.arange(train.n_items), train.seen_items(0))
    np.testing.assert_array_equal(first[0], np.sort(unseen)[:5])


def test_recommend_refuses_to_rank_nan_scores(split):
    """A diverged fit must fail loudly instead of returning arbitrary order."""
    _, train, _ = split
    model = NaNRecommender().fit(train)
    with pytest.raises(ValueError, match="NaN scores"):
        model.recommend(np.array([0]), 5)


def test_recommend_rejects_k_larger_than_the_catalogue(split):
    _, train, _ = split
    model = PopularityRecommender().fit(train)
    with pytest.raises(ValueError, match="exceeds"):
        model.recommend(np.array([0]), train.n_items + 1)


def test_unfitted_model_refuses_to_recommend():
    with pytest.raises(RuntimeError, match="has not been fitted"):
        PopularityRecommender().recommend(np.array([0]), 5)


def test_popularity_ranks_by_customer_count(split):
    _, train, _ = split
    model = PopularityRecommender().fit(train)
    scores = model.score(np.array([0, 1]))
    np.testing.assert_array_equal(scores[0], train.item_popularity())
    np.testing.assert_array_equal(scores[0], scores[1])


def test_recommendation_shape_and_ordering(split):
    _, train, _ = split
    model = PureSVDRecommender(n_factors=4).fit(train)
    rows = np.array([0, 1, 2])
    recommended = model.recommend(rows, 4)
    assert recommended.shape == (3, 4)
    scores = model.score(rows)
    for i in range(3):
        ordered = scores[i][recommended[i]]
        assert np.all(np.diff(ordered) <= 1e-12)


@pytest.mark.parametrize(
    "build",
    [lambda: ItemKNNRecommender(n_neighbours=None), lambda: PureSVDRecommender(n_factors=4)],
    ids=["ItemKNN", "PureSVD"],
)
def test_co_purchase_models_recover_the_community_structure(split, build):
    """Each customer's held-out product is the one their community shares.

    The fixture makes the answer unambiguous, so anything that models
    co-purchase should put it first for every single customer.
    """
    _, train, relevant = split
    model = build().fit(train)

    rows = np.array(sorted(relevant))
    recommended = model.recommend(rows, 1)
    hits = build_hits(recommended, [relevant[int(r)] for r in rows])
    assert hits.mean() == 1.0


def test_popularity_fails_where_co_purchase_succeeds(split):
    """Community B is larger, so its products outrank what an A customer wants.

    This is the whole reason the baseline is in the comparison: an accuracy
    number without it cannot distinguish a recommender from a bestseller list.
    """
    _, train, relevant = split
    model = PopularityRecommender().fit(train)

    rows = np.array(sorted(relevant))
    recommended = model.recommend(rows, 5)
    hits = build_hits(recommended, [relevant[int(r)] for r in rows])

    community_a = [i for i, r in enumerate(rows) if train.user_ids[r] < 2000]
    assert len(community_a) == 12
    assert hits[community_a].sum() == 0.0


def test_itemknn_zeroes_its_own_diagonal(split):
    """Self-similarity is 1 by definition and carries no recommendation."""
    _, train, _ = split
    model = ItemKNNRecommender(n_neighbours=None).fit(train)
    assert np.diag(model.similarity_).sum() == 0.0


def test_itemknn_neighbour_truncation_drops_the_weakest_links():
    """Needs distinct similarities, which the community fixture does not have.

    In the toy communities every in-community pair scores exactly the same, so
    a threshold cut keeps all of them: with ties at the boundary, asking for
    two neighbours legitimately returns five. Here the overlaps differ, so the
    truncation has something to bite on.
    """
    from scipy import sparse

    from recengine.data import Interactions

    # Item 0 shares 3 customers with item 1, 2 with item 2, 1 with item 3.
    rows = sparse.csr_matrix(
        np.array(
            [
                [1, 1, 1, 1],
                [1, 1, 1, 0],
                [1, 1, 0, 0],
                [1, 0, 0, 0],
            ],
            dtype=np.float32,
        )
    )
    train = Interactions(
        matrix=rows,
        user_ids=np.arange(4),
        item_ids=np.array(["i0", "i1", "i2", "i3"]),
    )

    full = ItemKNNRecommender(n_neighbours=None).fit(train)
    truncated = ItemKNNRecommender(n_neighbours=1).fit(train)

    assert (truncated.similarity_ != 0).sum() < (full.similarity_ != 0).sum()
    # Item 3's single strongest neighbour is item 2, not item 0.
    kept = np.flatnonzero(truncated.similarity_[3])
    assert kept.tolist() == [2]


def test_default_lineup_always_has_the_three_core_models():
    names = [m.name for m in default_models(include_legacy=False)]
    assert names == ["Popularity", "ItemKNN", "PureSVD"]


def test_default_lineup_excludes_the_unrankable_legacy_variant():
    """Only the clipped legacy model is in the lineup.

    The unclipped one diverges to NaN on the real target, so there is nothing
    to rank; that is recorded through the `diverged` property instead.
    """
    names = [m.name for m in default_models(include_legacy=True)]
    assert names == ["Popularity", "ItemKNN", "PureSVD", "LegacySVD"]
    assert "LegacySVD (unclipped)" not in names


def test_legacy_is_included_exactly_when_surprise_is_installed():
    """The evaluation must degrade to three models, not fail, without the extra.

    scikit-surprise is an optional extra with no wheel for every interpreter.
    A comparison that hard-required it would fail everywhere it is missing
    rather than simply running one model short.
    """
    from recengine.models import legacy_available

    has_legacy = any(m.name == "LegacySVD" for m in default_models())
    assert has_legacy == legacy_available()


# --- the legacy baseline -----------------------------------------------------


def _legacy(**kwargs):
    pytest.importorskip("surprise", reason="scikit-surprise is an optional extra")
    from recengine.models import LegacySVDRecommender

    return LegacySVDRecommender(**kwargs)


def test_legacy_requires_the_raw_transactions(split):
    _, train, _ = split
    model = _legacy()
    with pytest.raises(ValueError, match="needs the raw transactions"):
        model.fit(train, None)


def test_legacy_bulk_scoring_matches_surprise_predict(split):
    """The vectorised estimate has to agree with Surprise's own predict().

    NaN is compared explicitly. An earlier version of this check used
    abs(a - b) > tol, which is False when both sides are NaN, so it passed
    while every value on both sides was NaN.
    """
    train_frame, train, _ = split
    model = _legacy(random_state=0, clip=True).fit(train, train_frame)
    scores = model.score(np.arange(train.n_users))

    for row in range(train.n_users):
        for column in range(train.n_items):
            expected = model.algo_.predict(
                int(train.user_ids[row]), str(train.item_ids[column])
            ).est
            actual = scores[row, column]
            if np.isnan(expected):
                assert np.isnan(actual)
            else:
                assert actual == pytest.approx(expected, abs=1e-9)


def test_legacy_scores_stay_inside_the_declared_scale(split):
    train_frame, train, _ = split
    clipped = _legacy(random_state=0, clip=True).fit(train, train_frame)
    scores = clipped.score(np.arange(train.n_users))
    assert not np.isnan(scores).any()
    assert (scores >= 0).all() and (scores <= 1).all()


def test_legacy_clipping_maps_nan_to_the_upper_bound(split):
    """Surprise clips with Python's min/max, and min(1, nan) returns 1.

    Reproducing that quirk is the point of the model: it is what let a fit
    whose every parameter is NaN report a confident in-range prediction of
    exactly 1.0. numpy.clip would propagate the NaN instead.

    The NaN is injected rather than waited for. On the real data this fit
    diverges on its own, but the toy fixture is far too small and well behaved
    to blow up, so relying on divergence here would leave the branch untested.
    """
    train_frame, train, _ = split
    rows = np.arange(train.n_users)

    clipped = _legacy(random_state=0, clip=True).fit(train, train_frame)
    clipped.item_bias_[:] = np.nan
    scores = clipped.score(rows)
    assert not np.isnan(scores).any()
    assert (scores == 1.0).all()

    raw = _legacy(random_state=0, clip=False).fit(train, train_frame)
    raw.item_bias_[:] = np.nan
    assert np.isnan(raw.score(rows)).all()


def test_legacy_reports_whether_it_diverged(split):
    train_frame, train, _ = split
    model = _legacy(random_state=0).fit(train, train_frame)
    assert isinstance(model.diverged, bool)
