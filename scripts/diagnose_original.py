"""Reproduce and measure the behaviour of the original implementation.

This script is deliberately self-contained: it inlines the preprocessing and
model code exactly as they appeared in the first version of ``app.py``, so it
keeps working as a historical record no matter how the package around it
changes. Every claim the README makes about the original approach is produced
here, and the numbers are written to ``results/original_diagnosis.json`` so
they can be diffed rather than trusted.

Run from the repository root:

    python scripts/diagnose_original.py

Takes a few minutes, most of it in the final personalisation check, which
calls ``algo.predict`` once per candidate product exactly as the original
endpoint did.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data.csv"
OUT_PATH = ROOT / "results" / "original_diagnosis.json"

# The original hyperparameters, unchanged.
SVD_PARAMS = dict(n_factors=50, n_epochs=20, lr_all=0.005, reg_all=0.02)
TEST_SIZE = 0.25
SEEDS = [0, 1, 2, 3, 4]


def load_as_original() -> pd.DataFrame:
    """The preprocessing from the original ``load_and_preprocess_data``."""
    data = pd.read_csv(DATA_PATH, encoding="ISO-8859-1")
    data = data.dropna()
    data = data[(data["Quantity"] > 0) & (data["UnitPrice"] > 0)]
    data["InvoiceNo"] = data["InvoiceNo"].astype("str")
    data["StockCode"] = data["StockCode"].astype("str")
    scaler = StandardScaler()
    data[["Quantity", "UnitPrice"]] = scaler.fit_transform(
        data[["Quantity", "UnitPrice"]]
    )
    return data


def target_vs_declared_scale(data: pd.DataFrame) -> dict:
    """The target is a z-score; the code declares it lives in [0, 1]."""
    q = data["Quantity"].to_numpy()
    return {
        "n_rows": int(len(q)),
        "min": float(q.min()),
        "median": float(np.median(q)),
        "max": float(q.max()),
        "share_inside_declared_0_1": float(((q >= 0) & (q <= 1)).mean()),
        "share_below_zero": float((q < 0).mean()),
    }


def split_leakage(data: pd.DataFrame) -> dict:
    """A row-level random split puts the same (user, item) pair on both sides.

    The original hands Surprise one row per transaction line, so a customer who
    bought the same product on five invoices contributes five "ratings".
    """
    key = data["CustomerID"].astype(str) + "\x00" + data["StockCode"].astype(str)
    pair = pd.factorize(key)[0]
    n_rows = len(pair)
    counts = np.bincount(pair)

    rates = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        order = rng.permutation(n_rows)
        n_test = int(round(TEST_SIZE * n_rows))
        test_idx, train_idx = order[:n_test], order[n_test:]
        in_train = np.zeros(counts.size, dtype=bool)
        in_train[pair[train_idx]] = True
        rates.append(float(in_train[pair[test_idx]].mean()))

    return {
        "n_rows": int(n_rows),
        "n_distinct_pairs": int(counts.size),
        "share_rows_in_a_repeated_pair": float(counts[counts > 1].sum() / n_rows),
        # A test row leaks unless every other occurrence of its pair also
        # landed in test, which has probability about TEST_SIZE ** (n - 1).
        "analytic_leak_rate": float((1.0 - TEST_SIZE ** (counts[pair] - 1)).mean()),
        "empirical_leak_rate_mean": float(np.mean(rates)),
        "empirical_leak_rate_sd": float(np.std(rates)),
        "n_seeds": len(SEEDS),
    }


def fit_original(data: pd.DataFrame, seed: int):
    ds = Dataset.load_from_df(
        data[["CustomerID", "StockCode", "Quantity"]], Reader(rating_scale=(0, 1))
    )
    trainset, testset = train_test_split(ds, test_size=TEST_SIZE, random_state=seed)
    algo = SVD(random_state=seed, **SVD_PARAMS)
    algo.fit(trainset)
    return algo, trainset, testset


def accuracy_across_seeds(data: pd.DataFrame) -> list[dict]:
    """RMSE per seed, next to the RMSE of simply predicting the train mean.

    Also records how much of the prediction vector sits on a clip bound.
    Surprise clamps estimates to the declared rating_scale, so an estimate of
    exactly 0.0 or exactly 1.0 means the model wanted to go outside it.
    """
    rows = []
    for seed in SEEDS:
        algo, trainset, testset = fit_original(data, seed)
        preds = algo.test(testset)
        est = np.array([p.est for p in preds])
        true = np.array([p.r_ui for p in preds])
        rows.append(
            {
                "seed": seed,
                "rmse": float(accuracy.rmse(preds, verbose=False)),
                "mae": float(accuracy.mae(preds, verbose=False)),
                "constant_predictor_rmse": float(
                    np.sqrt(np.mean((true - trainset.global_mean) ** 2))
                ),
                "share_pinned_at_upper_bound": float((est >= 1 - 1e-9).mean()),
                "share_pinned_at_lower_bound": float((est <= 1e-9).mean()),
                "n_distinct_estimates": int(np.unique(np.round(est, 9)).size),
                "share_targets_outside_declared_scale": float(
                    ((true < 0) | (true > 1)).mean()
                ),
            }
        )
    return rows


def personalisation_check(data: pd.DataFrame, n_users: int = 8) -> dict:
    """Run the original /recommend body and see if two users ever differ.

    The original sorted predictions by ``.est`` and took the first ten. Python's
    sort is stable, so identical estimates leave the candidate order untouched
    and the "top 10" degenerates to the first ten unseen products in pivot
    column order, which is alphabetical by StockCode.
    """
    user_product_matrix = pd.pivot_table(
        data,
        index="CustomerID",
        columns="StockCode",
        values="Quantity",
        aggfunc="sum",
    ).fillna(0)

    algo, _, _ = fit_original(data, SEEDS[0])

    lists, est_spans = {}, []
    for user_id in user_product_matrix.index[:n_users]:
        user_ratings = user_product_matrix.loc[user_id]
        unseen = user_ratings[user_ratings == 0].index.tolist()
        recs = [algo.predict(user_id, product) for product in unseen]
        recs = sorted(recs, key=lambda x: x.est, reverse=True)[:10]
        lists[user_id] = [r.iid for r in recs]
        est_spans.append(float(max(r.est for r in recs) - min(r.est for r in recs)))

    first_user = user_product_matrix.index[0]
    row = user_product_matrix.loc[first_user]
    alphabetical_first_10 = row[row == 0].index.tolist()[:10]

    purchased = data.groupby(["CustomerID", "StockCode"])["Quantity"].sum()
    cells = user_product_matrix.to_numpy()

    return {
        "n_users_probed": int(n_users),
        "n_distinct_top10_lists": len({tuple(v) for v in lists.values()}),
        "max_estimate_spread_within_a_top10": float(max(est_spans)),
        "example_top10": lists[first_user],
        "equals_alphabetically_first_unseen": lists[first_user]
        == alphabetical_first_10,
        # The zero-fill also inverts the signal it is supposed to encode.
        "pivot_shape": list(cells.shape),
        "share_pivot_cells_exactly_zero": float((cells == 0).mean()),
        "median_scaled_quantity_of_a_real_purchase": float(purchased.median()),
        "share_real_purchases_below_the_zero_fill": float((purchased < 0).mean()),
    }


def main() -> None:
    started = time.time()
    print(f"reading {DATA_PATH.name}")
    data = load_as_original()
    print(f"  {len(data):,} rows after the original cleaning\n")

    report: dict = {}

    report["target_vs_declared_scale"] = target_vs_declared_scale(data)
    t = report["target_vs_declared_scale"]
    print("target after StandardScaler, against the declared rating_scale=(0, 1):")
    print(f"  range [{t['min']:.3f}, {t['max']:.3f}], median {t['median']:.3f}")
    print(f"  inside [0, 1]: {t['share_inside_declared_0_1']:.1%}")
    print(f"  below zero   : {t['share_below_zero']:.1%}\n")

    report["split_leakage"] = split_leakage(data)
    lk = report["split_leakage"]
    print("leakage of the random row-level split:")
    print(f"  rows in a repeated pair: {lk['share_rows_in_a_repeated_pair']:.1%}")
    print(f"  analytic leak rate     : {lk['analytic_leak_rate']:.2%}")
    print(
        f"  empirical leak rate    : {lk['empirical_leak_rate_mean']:.2%} "
        f"(sd {lk['empirical_leak_rate_sd']:.2%}, {lk['n_seeds']} seeds)\n"
    )

    report["accuracy_across_seeds"] = accuracy_across_seeds(data)
    print("accuracy, and what a constant predictor would have scored:")
    for r in report["accuracy_across_seeds"]:
        print(
            f"  seed {r['seed']}  RMSE {r['rmse']:.4f}  vs constant "
            f"{r['constant_predictor_rmse']:.4f}   pinned high "
            f"{r['share_pinned_at_upper_bound']:.1%}  pinned low "
            f"{r['share_pinned_at_lower_bound']:.1%}  distinct estimates "
            f"{r['n_distinct_estimates']}"
        )
    beaten = sum(
        r["rmse"] > r["constant_predictor_rmse"] for r in report["accuracy_across_seeds"]
    )
    print(f"  seeds where the constant predictor wins: {beaten}/{len(SEEDS)}\n")

    report["personalisation"] = personalisation_check(data)
    p = report["personalisation"]
    print("does the endpoint personalise?")
    print(f"  {p['n_users_probed']} users -> {p['n_distinct_top10_lists']} distinct list(s)")
    print(f"  widest score spread inside a top 10: {p['max_estimate_spread_within_a_top10']:.2e}")
    print(f"  list equals the alphabetically first unseen products: "
          f"{p['equals_alphabetically_first_unseen']}")
    print(f"  example: {p['example_top10']}\n")
    print("the zero fill, which is also the 'unseen' test:")
    print(f"  median scaled quantity of a real purchase: "
          f"{p['median_scaled_quantity_of_a_real_purchase']:+.4f}")
    print(f"  real purchases sitting below the 0 fill  : "
          f"{p['share_real_purchases_below_the_zero_fill']:.1%}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {OUT_PATH.relative_to(ROOT)}  ({time.time() - started:.0f}s)")


if __name__ == "__main__":
    main()
