"""Score every recommender on the same temporal split and write the results.

Produces the numbers the README quotes:

``results/metrics.csv``
    One row per model and cutoff k.
``results/evaluation_run.json``
    The configuration and the shape of the data it ran on, so a table can never
    be read without knowing what produced it.
``results/cutoff_sensitivity.csv``
    The same comparison at three different split dates. One split date can
    always flatter one model; if the ordering survives moving the boundary, it
    is a property of the models rather than of the calendar.
``results/figures/``
    Charts, if matplotlib is installed.

Run from the repository root:

    python scripts/evaluate.py
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from recengine import __version__  # noqa: E402
from recengine.data import (  # noqa: E402
    DEFAULT_CUTOFF,
    build_interactions,
    heldout_items,
    load_transactions,
    temporal_split,
)
from recengine.evaluate import DEFAULT_KS, evaluate_model  # noqa: E402
from recengine.models import LegacySVDRecommender, default_models  # noqa: E402

RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"

#: Three split dates spanning the last two months of trading.
SWEEP_CUTOFFS = ("2011-10-09", "2011-11-09", "2011-11-24")


def run_comparison(
    frame: pd.DataFrame,
    cutoff: str,
    ks: tuple[int, ...],
    seed: int,
) -> tuple[pd.DataFrame, dict]:
    """Fit and score every model at one split date."""
    train_frame, test_frame = temporal_split(frame, cutoff)
    train = build_interactions(train_frame)
    relevant = heldout_items(test_frame, train)

    if not relevant:
        raise ValueError(f"cutoff {cutoff} leaves no evaluable customers")

    tables, diverged = [], {}
    for model in default_models(random_state=seed):
        started = time.perf_counter()
        model.fit(train, train_frame)
        fit_seconds = time.perf_counter() - started

        if isinstance(model, LegacySVDRecommender):
            diverged[model.name] = model.diverged

        evaluation = evaluate_model(model, train, relevant, ks, fit_seconds)
        tables.append(evaluation.to_frame())
        row = evaluation.table[evaluation.table["k"] == 10].iloc[0]
        print(
            f"  {model.name:<12s} fit {fit_seconds:6.2f}s  "
            f"P@10 {row['precision']:.4f}  NDCG@10 {row['ndcg']:.4f}  "
            f"hit {row['hit_rate']:.3f}  coverage {row['coverage']:.3f}"
        )

    sizes = np.array([len(v) for v in relevant.values()])
    context = {
        "cutoff": cutoff,
        "train_rows": int(len(train_frame)),
        "test_rows": int(len(test_frame)),
        "train_users": train.n_users,
        "train_items": train.n_items,
        "train_interactions": int(train.matrix.nnz),
        "train_density": round(train.density, 6),
        "evaluable_users": int(len(relevant)),
        "relevant_per_user_median": float(np.median(sizes)),
        "relevant_per_user_mean": round(float(sizes.mean()), 2),
        "legacy_diverged": diverged,
    }
    return pd.concat(tables, ignore_index=True), context


def make_figures(metrics: pd.DataFrame, context: dict) -> list[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not installed, skipping figures")
        return []

    FIGURES.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    at_ten = metrics[metrics["k"] == 10].set_index("model")
    order = at_ten["ndcg"].sort_values(ascending=False).index.tolist()
    subtitle = (
        f"{context['evaluable_users']:,} customers, split at {context['cutoff']}, "
        f"{context['train_items']:,} products"
    )

    # 1. The headline: accuracy at k=10, every model on one axis.
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for axis, column, label in zip(
        axes,
        ["precision", "ndcg", "hit_rate"],
        ["Precision@10", "NDCG@10", "Hit rate@10"],
        strict=True,
    ):
        values = at_ten.loc[order, column]
        axis.bar(range(len(order)), values, color="#4C72B0")
        axis.set_xticks(range(len(order)))
        axis.set_xticklabels(order, rotation=20, ha="right")
        axis.set_title(label)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Top-10 accuracy by model\n{subtitle}", y=1.04)
    fig.tight_layout()
    path = FIGURES / "accuracy_at_10.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    written.append(path)

    # 2. Accuracy against catalogue coverage. A recommender can score by
    #    pushing bestsellers, and this is where that shows up.
    fig, axis = plt.subplots(figsize=(6.5, 5))
    for name in order:
        row = at_ten.loc[name]
        axis.scatter(row["coverage"], row["ndcg"], s=120)
        axis.annotate(
            name,
            (row["coverage"], row["ndcg"]),
            textcoords="offset points",
            xytext=(8, 4),
        )
    axis.set_xlabel("Catalogue coverage@10 (share of products ever recommended)")
    axis.set_ylabel("NDCG@10")
    axis.set_title(f"Accuracy against catalogue coverage\n{subtitle}")
    axis.spines[["top", "right"]].set_visible(False)
    # The labels are drawn to the upper right of their points, so the default
    # limits clip the rightmost one.
    axis.margins(x=0.22, y=0.14)
    fig.tight_layout()
    path = FIGURES / "accuracy_vs_coverage.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    written.append(path)

    # 3. How the ordering behaves as the list gets longer.
    fig, axis = plt.subplots(figsize=(6.5, 4.5))
    for name in order:
        series = metrics[metrics["model"] == name].sort_values("k")
        axis.plot(series["k"], series["ndcg"], marker="o", label=name)
    axis.set_xlabel("k")
    axis.set_ylabel("NDCG@k")
    axis.set_title(f"NDCG against list length\n{subtitle}")
    axis.legend(frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path = FIGURES / "ndcg_by_k.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    written.append(path)

    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cutoff", default=DEFAULT_CUTOFF)
    parser.add_argument("--ks", type=int, nargs="+", default=list(DEFAULT_KS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-sweep", action="store_true")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()

    ks = tuple(sorted(args.ks))
    RESULTS.mkdir(parents=True, exist_ok=True)

    print("loading data.csv")
    frame = load_transactions()
    print(f"  {len(frame):,} usable transaction rows\n")

    print(f"comparison at cutoff {args.cutoff}")
    metrics, context = run_comparison(frame, args.cutoff, ks, args.seed)
    metrics.to_csv(RESULTS / "metrics.csv", index=False)
    print(f"\nwrote results/metrics.csv ({len(metrics)} rows)")

    run = {
        "recengine_version": __version__,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "seed": args.seed,
        "ks": list(ks),
        **context,
    }
    (RESULTS / "evaluation_run.json").write_text(
        json.dumps(run, indent=2) + "\n", encoding="utf-8"
    )
    print("wrote results/evaluation_run.json")

    if not args.no_sweep:
        print("\ncutoff sensitivity")
        frames = []
        for cutoff in SWEEP_CUTOFFS:
            print(f"  cutoff {cutoff}")
            table, ctx = run_comparison(frame, cutoff, ks, args.seed)
            table = table[table["k"] == 10].copy()
            table.insert(1, "cutoff", cutoff)
            table.insert(2, "evaluable_users", ctx["evaluable_users"])
            frames.append(table)
        sweep = pd.concat(frames, ignore_index=True)
        sweep.to_csv(RESULTS / "cutoff_sensitivity.csv", index=False)
        print(f"\nwrote results/cutoff_sensitivity.csv ({len(sweep)} rows)")

    if not args.no_figures:
        print("\nfigures")
        for path in make_figures(metrics, context):
            print(f"  wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
