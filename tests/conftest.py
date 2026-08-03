"""Shared fixtures.

The toy dataset is built so that the correct answer follows from its
construction rather than from whatever the code happens to produce, and so
that a popularity baseline gets it *wrong*. Without that second property a
fixture cannot tell a recommender apart from a bestseller list, which is the
exact confusion this project exists to resolve.

Two communities of customers that never buy across the boundary:

* community A: 12 customers, products A0 to A5
* community B: 30 customers, products B0 to B5

Every customer buys all six of their own community's products except one, and
that one is what they buy in the held-out window. Because community B is larger,
its products are bought by 25 customers each against community A's 10, so
ranking by popularity puts all six B products above the single A product an A
customer actually wants. Anything that models co-purchase gets it first.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

TRAIN_DATE = "1/15/2011 10:00"
TEST_DATE = "6/15/2011 10:00"
CUTOFF = "2011-03-01"

COMMUNITY_A_CUSTOMERS = 12
COMMUNITY_B_CUSTOMERS = 30
PRODUCTS_PER_COMMUNITY = 6


def _community_rows(prefix: str, customer_ids: range, invoice: list[int]) -> list[dict]:
    rows = []
    products = [f"{prefix}{i}" for i in range(PRODUCTS_PER_COMMUNITY)]
    for position, customer in enumerate(customer_ids):
        held_out = products[position % PRODUCTS_PER_COMMUNITY]
        for product in products:
            invoice[0] += 1
            rows.append(
                {
                    "InvoiceNo": str(invoice[0]),
                    "StockCode": product,
                    "Description": f"PRODUCT {product}",
                    "Quantity": 2,
                    "InvoiceDate": TEST_DATE if product == held_out else TRAIN_DATE,
                    "UnitPrice": 1.5,
                    "CustomerID": float(customer),
                    "Country": "United Kingdom",
                }
            )
    return rows


@pytest.fixture
def toy_frame() -> pd.DataFrame:
    """The raw transaction log, before any cleaning."""
    invoice = [500000]
    rows = _community_rows("A", range(1000, 1000 + COMMUNITY_A_CUSTOMERS), invoice)
    rows += _community_rows("B", range(2000, 2000 + COMMUNITY_B_CUSTOMERS), invoice)
    return pd.DataFrame(rows)


@pytest.fixture
def toy_csv(tmp_path, toy_frame) -> str:
    """The same log with the junk a real extract carries, written to disk.

    One row of each kind the loader is supposed to drop, so the tests that
    check the cleaning have something to remove.
    """
    junk = pd.DataFrame(
        [
            # No customer id: unusable for collaborative filtering.
            {
                "InvoiceNo": "900001", "StockCode": "A0", "Description": "PRODUCT A0",
                "Quantity": 3, "InvoiceDate": TRAIN_DATE, "UnitPrice": 1.5,
                "CustomerID": np.nan, "Country": "United Kingdom",
            },
            # A cancellation.
            {
                "InvoiceNo": "C900002", "StockCode": "A1", "Description": "PRODUCT A1",
                "Quantity": -3, "InvoiceDate": TRAIN_DATE, "UnitPrice": 1.5,
                "CustomerID": 1000.0, "Country": "United Kingdom",
            },
            # A zero-price adjustment.
            {
                "InvoiceNo": "900003", "StockCode": "A2", "Description": "PRODUCT A2",
                "Quantity": 3, "InvoiceDate": TRAIN_DATE, "UnitPrice": 0.0,
                "CustomerID": 1000.0, "Country": "United Kingdom",
            },
            # Postage, which is a service line and not a product.
            {
                "InvoiceNo": "900004", "StockCode": "POST", "Description": "POSTAGE",
                "Quantity": 1, "InvoiceDate": TRAIN_DATE, "UnitPrice": 18.0,
                "CustomerID": 1000.0, "Country": "United Kingdom",
            },
            # A repeat of a pair that already exists, on another invoice.
            {
                "InvoiceNo": "900005", "StockCode": "A1", "Description": "PRODUCT A1",
                "Quantity": 7, "InvoiceDate": TRAIN_DATE, "UnitPrice": 1.5,
                "CustomerID": 1000.0, "Country": "United Kingdom",
            },
            # Customer 1000 buys A1 again after the cutoff. A1 is already in
            # their training history, so it must not count as something the
            # recommender was supposed to predict: the models exclude products
            # the customer already has, so scoring it would count an
            # unreachable hit as a miss.
            {
                "InvoiceNo": "900006", "StockCode": "A1", "Description": "PRODUCT A1",
                "Quantity": 4, "InvoiceDate": TEST_DATE, "UnitPrice": 1.5,
                "CustomerID": 1000.0, "Country": "United Kingdom",
            },
        ]
    )
    path = tmp_path / "toy.csv"
    pd.concat([toy_frame, junk], ignore_index=True).to_csv(
        path, index=False, encoding="ISO-8859-1"
    )
    return str(path)
