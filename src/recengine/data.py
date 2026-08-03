"""Loading, cleaning and splitting the Online Retail transaction log.

The source file is a transaction log, not a ratings table. There is no column
anywhere in it that says how much a customer liked a product; there is only
evidence that a purchase happened. That is *implicit feedback*, and it drives
every decision in this module:

* the interaction matrix is binary, "this customer bought this product at least
  once", rather than a quantity. A quantity is a property of the basket, not of
  preference, and it is dominated by wholesale orders: one row in this file has
  a quantity of 80,995.
* rows are deduplicated to one per (customer, product). Buying the same item on
  six invoices is one preference observation repeated, not six.
* the split is by time, not at random. A random split over a year of trading
  lets a model train on next month to predict last month.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = ROOT / "data.csv"

#: The published file uses this encoding; it is not UTF-8.
ENCODING = "ISO-8859-1"

#: The last invoice is dated 2011-12-09, so this holds out the final month.
DEFAULT_CUTOFF = "2011-11-09"

#: Codes that bill for a service rather than identify a product. They are only
#: 0.39% of rows, but POSTAGE alone appears on 1,099 of them, which is enough
#: to put it near the top of a popularity ranking. Recommending postage to a
#: customer is not a recommendation.
SERVICE_CODES = frozenset({"POST", "M", "C2", "DOT", "BANK CHARGES", "PADS"})


@dataclass(frozen=True)
class Interactions:
    """A binary user-item matrix plus the labels for its rows and columns.

    ``matrix`` is CSR of shape (n_users, n_items) holding 1.0 where the customer
    bought the product inside the window this was built from.
    """

    matrix: sparse.csr_matrix
    user_ids: np.ndarray
    item_ids: np.ndarray

    def __post_init__(self) -> None:
        if self.matrix.shape != (len(self.user_ids), len(self.item_ids)):
            raise ValueError(
                f"matrix shape {self.matrix.shape} does not match "
                f"{len(self.user_ids)} users x {len(self.item_ids)} items"
            )

    @property
    def n_users(self) -> int:
        return len(self.user_ids)

    @property
    def n_items(self) -> int:
        return len(self.item_ids)

    @property
    def density(self) -> float:
        return self.matrix.nnz / (self.n_users * self.n_items)

    @property
    def user_index(self) -> dict[int, int]:
        """CustomerID -> row number."""
        return {int(u): i for i, u in enumerate(self.user_ids)}

    @property
    def item_index(self) -> dict[str, int]:
        """StockCode -> column number."""
        return {str(s): j for j, s in enumerate(self.item_ids)}

    def item_popularity(self) -> np.ndarray:
        """Number of distinct customers who bought each product."""
        return np.asarray(self.matrix.sum(axis=0)).ravel()

    def seen_items(self, row: int) -> np.ndarray:
        """Column indices the user in ``row`` already bought."""
        start, end = self.matrix.indptr[row], self.matrix.indptr[row + 1]
        return self.matrix.indices[start:end]


def load_transactions(path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Read the transaction log and drop what cannot be used.

    Removals, in order, and why:

    * rows with any null field. Overwhelmingly this is a null ``CustomerID``
      (135,080 rows), which cannot be attributed to anyone and so is useless
      for collaborative filtering.
    * non-positive ``Quantity``, which marks returns and cancellations. Every
      row whose ``InvoiceNo`` starts with "C" is removed by this filter.
    * non-positive ``UnitPrice``, which marks adjustments rather than sales.
    * the service codes in :data:`SERVICE_CODES`.
    """
    frame = pd.read_csv(path, encoding=ENCODING)
    frame = frame.dropna()
    frame = frame[(frame["Quantity"] > 0) & (frame["UnitPrice"] > 0)]

    frame["StockCode"] = frame["StockCode"].astype(str).str.strip().str.upper()
    frame = frame[~frame["StockCode"].isin(SERVICE_CODES)]

    frame["CustomerID"] = frame["CustomerID"].astype(np.int64)
    frame["InvoiceNo"] = frame["InvoiceNo"].astype(str)
    frame["InvoiceDate"] = pd.to_datetime(frame["InvoiceDate"], format="%m/%d/%Y %H:%M")

    return frame.reset_index(drop=True)


def temporal_split(
    frame: pd.DataFrame, cutoff: str = DEFAULT_CUTOFF
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split on invoice date: everything before ``cutoff`` trains, the rest tests.

    This is the only split that answers the question the product actually asks,
    which is "given what this customer has bought so far, what will they buy
    next". A random split answers a question nobody has.
    """
    boundary = pd.Timestamp(cutoff)
    if not (frame["InvoiceDate"].min() < boundary < frame["InvoiceDate"].max()):
        raise ValueError(
            f"cutoff {cutoff} lies outside the data range "
            f"{frame['InvoiceDate'].min()} to {frame['InvoiceDate'].max()}"
        )
    before = frame["InvoiceDate"] < boundary
    return frame[before].copy(), frame[~before].copy()


def build_interactions(frame: pd.DataFrame) -> Interactions:
    """Build the binary user-item matrix from a window of transactions.

    Repeat purchases of the same product collapse to a single 1. Users and
    items are ordered by their sorted id so the matrix is reproducible.
    """
    pairs = frame[["CustomerID", "StockCode"]].drop_duplicates()

    user_ids = np.sort(pairs["CustomerID"].unique())
    item_ids = np.sort(pairs["StockCode"].unique().astype(str))

    rows = np.searchsorted(user_ids, pairs["CustomerID"].to_numpy())
    cols = np.searchsorted(item_ids, pairs["StockCode"].to_numpy().astype(str))

    matrix = sparse.csr_matrix(
        (np.ones(len(pairs), dtype=np.float32), (rows, cols)),
        shape=(len(user_ids), len(item_ids)),
    )
    matrix.sort_indices()
    return Interactions(matrix=matrix, user_ids=user_ids, item_ids=item_ids)


def heldout_items(
    test: pd.DataFrame, train: Interactions
) -> dict[int, np.ndarray]:
    """The evaluation target: what each known customer bought next, that is new.

    Restricted deliberately on three counts, each of which would otherwise
    flatter or distort the scores:

    * customers absent from training are dropped. Nothing personalised can be
      said about them, and leaving them in measures cold-start handling while
      pretending to measure ranking quality.
    * products absent from training are dropped. No model can rank a column it
      has never seen.
    * products the customer already bought during training are dropped, because
      the recommenders exclude already-seen items from their output, so leaving
      these in would count unreachable hits as misses.

    Returns a mapping from matrix row number to the column indices that count
    as relevant. Customers left with nothing relevant are omitted.
    """
    user_index = train.user_index
    item_index = train.item_index

    pairs = test[["CustomerID", "StockCode"]].drop_duplicates()
    rows = pairs["CustomerID"].map(user_index)
    cols = pairs["StockCode"].astype(str).map(item_index)

    known = rows.notna() & cols.notna()
    rows = rows[known].to_numpy(dtype=np.int64)
    cols = cols[known].to_numpy(dtype=np.int64)

    order = np.argsort(rows, kind="stable")
    rows, cols = rows[order], cols[order]

    relevant: dict[int, np.ndarray] = {}
    for row, group in zip(*_group_by_row(rows, cols), strict=True):
        novel = np.setdiff1d(group, train.seen_items(row), assume_unique=False)
        if novel.size:
            relevant[int(row)] = novel
    return relevant


def _group_by_row(
    rows: np.ndarray, cols: np.ndarray
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Split ``cols`` into one array per distinct value in sorted ``rows``."""
    unique, starts = np.unique(rows, return_index=True)
    return unique, np.split(cols, starts[1:])


def item_catalogue(frame: pd.DataFrame) -> pd.Series:
    """StockCode -> product name, for turning recommendations into English.

    213 stock codes carry more than one spelling of their description across the
    file, so the most frequent one wins rather than whichever row came first.
    """
    counts = (
        frame.groupby(["StockCode", "Description"], observed=True)
        .size()
        .reset_index(name="n")
        .sort_values(["StockCode", "n"], ascending=[True, False])
    )
    return counts.drop_duplicates("StockCode").set_index("StockCode")["Description"]
