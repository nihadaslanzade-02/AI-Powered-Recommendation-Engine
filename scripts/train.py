"""Fit a recommender on the whole transaction log and save it for serving.

The original trained inside ``app.py`` at import time, which meant every
container start paid for a full fit, every worker process fitted its own
independent copy, and the model a request was answered from depended on which
worker happened to pick it up. Training belongs here, once, ahead of time.

Run from the repository root:

    python scripts/train.py

Writes ``artifacts/model.joblib``. That path is gitignored: it is derived from
data.csv and rebuilding it takes under a second.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from recengine import __version__  # noqa: E402
from recengine.data import (  # noqa: E402
    build_interactions,
    item_catalogue,
    load_transactions,
)
from recengine.models import (  # noqa: E402
    ItemKNNRecommender,
    PopularityRecommender,
    PureSVDRecommender,
)

ARTIFACT_DIR = ROOT / "artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "model.joblib"

# PureSVD wins on every metric at every cutoff tested, see results/metrics.csv.
BUILDERS = {
    "puresvd": lambda seed: PureSVDRecommender(n_factors=50, random_state=seed),
    "itemknn": lambda seed: ItemKNNRecommender(n_neighbours=200),
    "popularity": lambda seed: PopularityRecommender(),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(BUILDERS), default="puresvd")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=ARTIFACT_PATH)
    args = parser.parse_args()

    print("loading data.csv")
    frame = load_transactions()
    print(f"  {len(frame):,} usable transaction rows")

    # Serving fits on everything. The held-out split exists to measure quality,
    # not to serve from, and throwing away the most recent month of behaviour
    # in production would be the opposite of what it was for.
    train = build_interactions(frame)
    catalogue = item_catalogue(frame)
    print(f"  {train.n_users:,} customers x {train.n_items:,} products, "
          f"{train.matrix.nnz:,} interactions")

    model = BUILDERS[args.model](args.seed)
    started = time.perf_counter()
    model.fit(train, frame)
    fit_seconds = time.perf_counter() - started
    print(f"fitted {model.name} in {fit_seconds:.2f}s")

    artifact = {
        "model": model,
        "catalogue": catalogue,
        "metadata": {
            "model": model.name,
            "recengine_version": __version__,
            "python": platform.python_version(),
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "seed": args.seed,
            "n_users": train.n_users,
            "n_items": train.n_items,
            "n_interactions": int(train.matrix.nnz),
            "fit_seconds": round(fit_seconds, 3),
            "data_rows": int(len(frame)),
            "first_invoice": str(frame["InvoiceDate"].min()),
            "last_invoice": str(frame["InvoiceDate"].max()),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.output, compress=3)
    size_mb = args.output.stat().st_size / 1024**2
    print(f"wrote {args.output.relative_to(ROOT)} ({size_mb:.1f} MB)")
    print(json.dumps(artifact["metadata"], indent=2))


if __name__ == "__main__":
    main()
