"""Loading, cleaning, the interaction matrix and the split."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from recengine.data import (
    build_interactions,
    heldout_items,
    item_catalogue,
    load_transactions,
    temporal_split,
)

from .conftest import CUTOFF


def test_drops_rows_without_a_customer(toy_csv):
    frame = load_transactions(toy_csv)
    assert frame["CustomerID"].notna().all()
    assert "900001" not in set(frame["InvoiceNo"])


def test_drops_cancellations_and_zero_price_adjustments(toy_csv):
    frame = load_transactions(toy_csv)
    assert (frame["Quantity"] > 0).all()
    assert (frame["UnitPrice"] > 0).all()
    assert not frame["InvoiceNo"].str.startswith("C").any()


def test_drops_service_codes(toy_csv):
    """POSTAGE is on enough invoices to rank high; it is not a product."""
    frame = load_transactions(toy_csv)
    assert "POST" not in set(frame["StockCode"])


def test_customer_id_is_an_integer(toy_csv):
    frame = load_transactions(toy_csv)
    assert frame["CustomerID"].dtype == np.int64


def test_temporal_split_puts_nothing_from_the_future_in_train(toy_csv):
    frame = load_transactions(toy_csv)
    train, test = temporal_split(frame, CUTOFF)
    boundary = pd.Timestamp(CUTOFF)
    assert (train["InvoiceDate"] < boundary).all()
    assert (test["InvoiceDate"] >= boundary).all()
    assert len(train) + len(test) == len(frame)


def test_temporal_split_rejects_a_cutoff_outside_the_data(toy_csv):
    frame = load_transactions(toy_csv)
    with pytest.raises(ValueError, match="outside the data range"):
        temporal_split(frame, "2050-01-01")


def test_interactions_are_binary_and_deduplicated(toy_csv):
    """A repeat purchase of the same product is one preference, not two.

    The fixture has customer 1000 buying A1 on two separate invoices. Left as
    separate rows, that is what let the original's random split place the same
    pair on both sides of the evaluation.
    """
    frame = load_transactions(toy_csv)
    train, _ = temporal_split(frame, CUTOFF)

    repeated = train[(train["CustomerID"] == 1000) & (train["StockCode"] == "A1")]
    assert len(repeated) == 2, "fixture should contain the duplicate pair"

    interactions = build_interactions(train)
    assert set(np.unique(interactions.matrix.data)) == {1.0}

    row = interactions.user_index[1000]
    column = interactions.item_index["A1"]
    assert interactions.matrix[row, column] == 1.0


def test_no_relevant_item_is_already_in_the_training_history(toy_csv):
    """The leakage guard, stated as the property that matters.

    Under the original protocol 43.70% of test rows were pairs the model had
    trained on. Here it has to be exactly zero.
    """
    frame = load_transactions(toy_csv)
    train_frame, test_frame = temporal_split(frame, CUTOFF)
    train = build_interactions(train_frame)
    relevant = heldout_items(test_frame, train)

    assert relevant, "fixture should leave evaluable customers"
    for row, items in relevant.items():
        overlap = np.intersect1d(items, train.seen_items(row))
        assert overlap.size == 0

    # The fixture has customer 1000 rebuying A1 after the cutoff, so this is
    # actually exercised rather than being vacuously true.
    row = train.user_index[1000]
    assert train.item_index["A1"] in set(train.seen_items(row))
    assert train.item_index["A1"] not in set(relevant[row])


def test_heldout_drops_customers_and_products_absent_from_training(toy_csv):
    frame = load_transactions(toy_csv)
    train_frame, test_frame = temporal_split(frame, CUTOFF)
    train = build_interactions(train_frame)

    stranger = test_frame.iloc[[0]].copy()
    stranger["CustomerID"] = 999999
    stranger["StockCode"] = "NEVER_SEEN"
    relevant = heldout_items(
        pd.concat([test_frame, stranger], ignore_index=True), train
    )

    assert 999999 not in train.user_index
    assert all(row < train.n_users for row in relevant)
    for items in relevant.values():
        assert (items < train.n_items).all()


def test_every_customer_holds_out_exactly_one_product(toy_csv):
    """The fixture is built that way, so the harness must see it that way."""
    frame = load_transactions(toy_csv)
    train_frame, test_frame = temporal_split(frame, CUTOFF)
    train = build_interactions(train_frame)
    relevant = heldout_items(test_frame, train)

    assert len(relevant) == 42
    assert {len(v) for v in relevant.values()} == {1}


def test_interaction_shape_matches_its_labels(toy_csv):
    frame = load_transactions(toy_csv)
    interactions = build_interactions(frame)
    assert interactions.matrix.shape == (interactions.n_users, interactions.n_items)
    assert interactions.n_users == 42
    assert interactions.n_items == 12


def test_item_popularity_counts_customers_not_rows(toy_csv):
    frame = load_transactions(toy_csv)
    train_frame, _ = temporal_split(frame, CUTOFF)
    train = build_interactions(train_frame)
    popularity = train.item_popularity()

    # Customer 1000 bought A1 twice; it must still count once.
    assert popularity[train.item_index["A1"]] == 10


def test_item_catalogue_prefers_the_most_common_description():
    frame = pd.DataFrame(
        {
            "StockCode": ["X", "X", "X", "Y"],
            "Description": ["COMMON NAME", "COMMON NAME", "typo", "OTHER"],
        }
    )
    catalogue = item_catalogue(frame)
    assert catalogue["X"] == "COMMON NAME"
    assert catalogue["Y"] == "OTHER"
