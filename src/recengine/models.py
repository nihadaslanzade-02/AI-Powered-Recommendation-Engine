"""The recommenders, all behind one interface.

Every model implements a single method, :meth:`Recommender.score`, returning a
score for each candidate product. Excluding products the customer already
bought, breaking ties and taking the top k all happen once in the base class,
so no model can come out ahead because of how its own shortlisting happened to
be written. That matters more than it sounds: the original returned the
alphabetically first unseen products precisely because tie-breaking was left
to a stable sort over identical scores.

The lineup is a ladder, and the point of the ladder is that each rung has to
justify the one below it:

``PopularityRecommender``
    Not personalised at all. If a personalised model cannot beat this, it is
    not earning its complexity.
``ItemKNNRecommender``
    Co-purchase similarity. The workhorse of retail recommendation.
``PureSVDRecommender``
    Truncated SVD of the binary matrix, following Cremonesi, Koren and Turrin,
    "Performance of recommender algorithms on top-N recommendation tasks",
    RecSys 2010, which is also the paper that showed RMSE-tuned models can lose
    to popularity once you actually measure the ranking.
``LegacySVDRecommender``
    The original approach, preserved: standardised Quantity as the target and a
    declared rating scale of (0, 1). Kept so the comparison shows what it
    scores rather than merely asserting that it was wrong.
"""

from __future__ import annotations

import importlib.util
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds

from .data import Interactions


class Recommender(ABC):
    """Base class holding the parts every model must share."""

    name: str = "recommender"

    def __init__(self) -> None:
        self.train: Interactions | None = None

    def fit(
        self, train: Interactions, transactions: pd.DataFrame | None = None
    ) -> Recommender:
        """Learn from the training window.

        ``transactions`` is the raw frame the matrix was built from. Only the
        legacy model needs it, because it trains on a quantity column that the
        binary matrix deliberately throws away; the rest ignore it.
        """
        self.train = train
        self._fit(train, transactions)
        return self

    @abstractmethod
    def _fit(self, train: Interactions, transactions: pd.DataFrame | None) -> None: ...

    @abstractmethod
    def score(self, rows: np.ndarray) -> np.ndarray:
        """Scores of shape (len(rows), n_items). Higher ranks first."""

    def recommend(
        self, rows: np.ndarray, k: int, exclude_seen: bool = True
    ) -> np.ndarray:
        """Top ``k`` product columns for each user row, best first.

        Ties are broken by ascending column index. Any deterministic rule would
        do; the requirement is that there *is* one, so that a model emitting
        constant scores produces an obviously arbitrary list rather than one
        that quietly looks sorted.

        This uses a full stable sort rather than ``argpartition``. Partitioning
        is the faster way to take a top k, but it only promises the k largest
        *values*: which of several equally scored products it hands back is
        unspecified, so a model with tied scores would return a different list
        from one call to the next. Sorting the whole row costs a few hundred
        milliseconds across the entire customer base, which is not worth
        trading for a ranking that cannot be reproduced.
        """
        if self.train is None:
            raise RuntimeError(f"{self.name} has not been fitted")
        rows = np.asarray(rows, dtype=np.int64)
        if k > self.train.n_items:
            raise ValueError(f"k={k} exceeds the {self.train.n_items} products available")

        scores = np.asarray(self.score(rows), dtype=np.float64)
        if np.isnan(scores).any():
            raise ValueError(
                f"{self.name} produced NaN scores, so there is no ranking to "
                "return. Training did not converge."
            )
        if exclude_seen:
            for i, row in enumerate(rows):
                scores[i, self.train.seen_items(row)] = -np.inf

        # Stable sort keeps equally scored products in column order, which is
        # ascending index, so the tie-breaking rule needs no extra key.
        return np.argsort(-scores, axis=1, kind="stable")[:, :k]


class PopularityRecommender(Recommender):
    """Rank by how many customers bought each product. Identical for everyone."""

    name = "Popularity"

    def _fit(self, train: Interactions, transactions: pd.DataFrame | None) -> None:
        self.popularity_ = train.item_popularity()

    def score(self, rows: np.ndarray) -> np.ndarray:
        return np.tile(self.popularity_, (len(rows), 1))


class ItemKNNRecommender(Recommender):
    """Item-item cosine similarity over the binary matrix.

    A customer's score for a candidate is the summed similarity between that
    candidate and everything they have already bought.

    ``n_neighbours`` truncates each product's similarity row to its closest
    neighbours. Without it every product is weakly similar to every other, and
    those many small terms accumulate into a popularity signal that drowns out
    the specific co-purchase evidence.
    """

    name = "ItemKNN"

    def __init__(self, n_neighbours: int | None = 200, shrink: float = 0.0) -> None:
        super().__init__()
        self.n_neighbours = n_neighbours
        self.shrink = shrink

    def _fit(self, train: Interactions, transactions: pd.DataFrame | None) -> None:
        matrix = train.matrix.astype(np.float64)
        co_occurrence = np.asarray((matrix.T @ matrix).todense())

        counts = np.diag(co_occurrence).copy()
        norms = np.sqrt(counts)
        denominator = np.outer(norms, norms) + self.shrink
        denominator[denominator == 0] = 1.0

        similarity = co_occurrence / denominator
        np.fill_diagonal(similarity, 0.0)

        if self.n_neighbours is not None and self.n_neighbours < similarity.shape[1]:
            cut = similarity.shape[1] - self.n_neighbours
            threshold = np.partition(similarity, cut, axis=1)[:, cut][:, None]
            similarity[similarity < threshold] = 0.0

        self.similarity_ = similarity

    def score(self, rows: np.ndarray) -> np.ndarray:
        assert self.train is not None
        return self.train.matrix[rows].astype(np.float64) @ self.similarity_


class PureSVDRecommender(Recommender):
    """Truncated SVD of the binary matrix, scored by folding the user row in.

    The score matrix is ``X V V^T``: project a customer's purchase history into
    the latent space and back out again. Because it works from the row rather
    than from a stored user factor, it scores customers who were not part of the
    factorisation, which is what the API needs when a new order arrives between
    retrains.
    """

    name = "PureSVD"

    def __init__(self, n_factors: int = 50, random_state: int = 0) -> None:
        super().__init__()
        self.n_factors = n_factors
        self.random_state = random_state

    def _fit(self, train: Interactions, transactions: pd.DataFrame | None) -> None:
        rng = np.random.default_rng(self.random_state)
        v0 = rng.standard_normal(min(train.matrix.shape))
        _, _, item_factors = svds(
            train.matrix.astype(np.float64), k=self.n_factors, v0=v0
        )
        # svds returns singular values ascending; orientation is arbitrary, but
        # V V^T is invariant to both, so no reordering is needed.
        self.item_factors_ = item_factors.T

    def score(self, rows: np.ndarray) -> np.ndarray:
        assert self.train is not None
        history = self.train.matrix[rows].astype(np.float64)
        return (history @ self.item_factors_) @ self.item_factors_.T


class LegacySVDRecommender(Recommender):
    """The original approach, preserved so it can be measured rather than asserted.

    Trains Surprise's biased SVD on standardised ``Quantity`` with a declared
    ``rating_scale`` of (0, 1), exactly as the first version of ``app.py`` did,
    including one row per transaction line rather than one per (customer,
    product) pair.

    Scoring reproduces ``AlgoBase.predict``: the biased estimate, falling back
    to the training mean for anything the model never saw, and then clipped to
    the declared scale.

    The clipping has to be reproduced exactly, quirk included, because it is
    what hides the real failure. On this target the SGD diverges: with a
    learning rate of 0.005 and residuals reaching 451, the updates overflow and
    every learned parameter ends up NaN. Surprise then clips with Python's
    builtin ``min`` and ``max``, and ``min(1, nan)`` returns 1, because ``nan <
    1`` is False and ``min`` falls through to its first argument. So a model
    that learned nothing at all reports a clean, in-range prediction of exactly
    1.0 and sets ``was_impossible`` to False.

    ``numpy.clip`` propagates NaN instead, so using it here would quietly make
    this model look broken in a different way than the deployed one was.
    ``clip=False`` exposes the raw estimate, which is how the comparison can
    tell "the clipping destroyed the ranking" apart from "there was no ranking
    information to destroy".
    """

    name = "LegacySVD"

    def __init__(
        self,
        n_factors: int = 50,
        n_epochs: int = 20,
        lr_all: float = 0.005,
        reg_all: float = 0.02,
        random_state: int = 0,
        clip: bool = True,
    ) -> None:
        super().__init__()
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr_all = lr_all
        self.reg_all = reg_all
        self.random_state = random_state
        self.clip = clip
        if not clip:
            self.name = "LegacySVD (unclipped)"

    def _fit(self, train: Interactions, transactions: pd.DataFrame | None) -> None:
        from sklearn.preprocessing import StandardScaler
        from surprise import SVD, Dataset, Reader

        if transactions is None:
            raise ValueError(
                "LegacySVDRecommender needs the raw transactions: it trains on "
                "standardised Quantity, which the binary matrix discards"
            )

        frame = transactions[["CustomerID", "StockCode", "Quantity"]].copy()
        frame["StockCode"] = frame["StockCode"].astype(str)
        frame["Quantity"] = StandardScaler().fit_transform(frame[["Quantity"]])

        dataset = Dataset.load_from_df(frame, Reader(rating_scale=(0, 1)))
        algo = SVD(
            n_factors=self.n_factors,
            n_epochs=self.n_epochs,
            lr_all=self.lr_all,
            reg_all=self.reg_all,
            random_state=self.random_state,
        )
        algo.fit(dataset.build_full_trainset())

        self.algo_ = algo
        self.global_mean_ = algo.trainset.global_mean
        self._build_index_maps(train, algo.trainset)

    def _build_index_maps(self, train: Interactions, trainset) -> None:
        """Map our matrix rows and columns onto Surprise's internal ids.

        Surprise keys its tables by the raw values it was handed, so ids are
        matched through ``str`` rather than by relying on both sides agreeing
        on numpy versus Python integer types.
        """
        self.user_inner_ = _inner_ids(train.user_ids, trainset, users=True)
        self.item_inner_ = _inner_ids(train.item_ids, trainset, users=False)

        # Item bias and factors are fixed at fit time, so build them once here
        # rather than on every scoring call.
        algo = self.algo_
        known = self.item_inner_ >= 0
        self.item_bias_ = np.zeros(train.n_items)
        self.item_bias_[known] = algo.bi[self.item_inner_[known]]
        self.item_factors_ = np.zeros((train.n_items, algo.qi.shape[1]))
        self.item_factors_[known] = algo.qi[self.item_inner_[known]]

    def score(self, rows: np.ndarray) -> np.ndarray:
        """Reproduce ``SVD.estimate`` for every product at once.

        Surprise's estimate is ``mu + b_u + b_i + q_i . p_u``, dropping whichever
        terms refer to something it never saw. Unknown items keep a zero bias
        and a zero factor row, which reproduces that fallback without a branch.
        """
        assert self.train is not None
        algo = self.algo_

        scores = np.tile(self.global_mean_ + self.item_bias_, (len(rows), 1))

        inner_users = self.user_inner_[rows]
        known = inner_users >= 0
        if known.any():
            scores[known] += algo.bu[inner_users[known]][:, None]
            scores[known] += algo.pu[inner_users[known]] @ self.item_factors_.T

        if self.clip:
            lower, higher = algo.trainset.rating_scale
            # Deliberately not np.clip: Surprise uses Python's min/max, which
            # map NaN to the upper bound rather than propagating it. See the
            # class docstring; reproducing this is the whole point of the model.
            scores = np.where(np.isnan(scores), higher, np.clip(scores, lower, higher))
        return scores

    @property
    def diverged(self) -> bool:
        """True when training produced NaN parameters rather than a model."""
        return bool(
            np.isnan(self.algo_.pu).any()
            or np.isnan(self.algo_.qi).any()
            or np.isnan(self.algo_.bu).any()
            or np.isnan(self.algo_.bi).any()
        )


def _inner_ids(raw_ids: np.ndarray, trainset, users: bool) -> np.ndarray:
    """Surprise's inner index for each raw id, or -1 where it is unknown."""
    lookup = (
        trainset._raw2inner_id_users if users else trainset._raw2inner_id_items
    )
    by_string = {str(raw): inner for raw, inner in lookup.items()}
    return np.array(
        [by_string.get(str(raw), -1) for raw in raw_ids], dtype=np.int64
    )


def legacy_available() -> bool:
    """Whether the optional Surprise dependency can be imported."""
    return importlib.util.find_spec("surprise") is not None


def default_models(
    random_state: int = 0, include_legacy: bool | None = None
) -> list[Recommender]:
    """The lineup used by ``scripts/evaluate.py``.

    ``LegacySVDRecommender`` is included with ``clip=True`` because that is what
    the deployed endpoint did. The unclipped variant is not: on this target the
    fit diverges, so its raw scores are NaN and there is nothing to rank. That
    is a finding rather than an obstacle, and it is recorded through the
    ``diverged`` property instead of through an unrankable extra row.

    Whether an SVD can work here at all is already answered by
    ``PureSVDRecommender``, which is the same family of model pointed at a
    target that means something.

    ``include_legacy`` defaults to whether Surprise is installed. It is an
    optional extra with no wheel for every interpreter, so a comparison that
    hard-required it would fail everywhere it is missing rather than simply
    running one model short. Pass ``True`` to insist on it and get the
    ImportError instead of a quiet omission.
    """
    if include_legacy is None:
        include_legacy = legacy_available()

    models: list[Recommender] = [
        PopularityRecommender(),
        ItemKNNRecommender(n_neighbours=200),
        PureSVDRecommender(n_factors=50, random_state=random_state),
    ]
    if include_legacy:
        models.append(LegacySVDRecommender(random_state=random_state, clip=True))
    return models
