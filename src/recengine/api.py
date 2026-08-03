"""The Flask service.

Two things are deliberately different from the original.

Nothing is fitted here. The app loads an artifact produced by
``scripts/train.py``; if it cannot, it still starts and reports itself
unhealthy, because a container that refuses to come up is harder to diagnose
than one that comes up and tells you what is wrong.

Scoring is one matrix product rather than a Python loop. The original called
``algo.predict`` once per candidate product, so a single request did roughly
3,600 sequential calls.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from flask import Flask, jsonify, request

from . import __version__

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT = ROOT / "artifacts" / "model.joblib"

#: Requests above this are rejected rather than silently truncated.
MAX_K = 100
DEFAULT_K = 10


class ModelStore:
    """Holds the loaded artifact, or the reason there isn't one."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.model: Any = None
        self.catalogue: Any = None
        self.metadata: dict = {}
        self.error: str | None = None
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            self.error = (
                f"no artifact at {self.path}. Build one with: python scripts/train.py"
            )
            return
        try:
            artifact = joblib.load(self.path)
            self.model = artifact["model"]
            self.catalogue = artifact["catalogue"]
            self.metadata = artifact["metadata"]
        except Exception as exc:  # noqa: BLE001 - reported through /health
            self.error = f"could not load {self.path}: {exc}"

    @property
    def ready(self) -> bool:
        return self.model is not None

    def row_for(self, user_id: int) -> int | None:
        index = self.model.train.user_index
        return index.get(int(user_id))

    def recommend(self, user_id: int, k: int) -> list[dict]:
        row = self.row_for(user_id)
        if row is None:
            raise KeyError(user_id)

        rows = np.array([row], dtype=np.int64)
        columns = self.model.recommend(rows, k)[0]
        scores = self.model.score(rows)[0]
        item_ids = self.model.train.item_ids

        out = []
        for rank, column in enumerate(columns, start=1):
            stock_code = str(item_ids[column])
            out.append(
                {
                    "rank": rank,
                    "stock_code": stock_code,
                    "description": _describe(self.catalogue, stock_code),
                    "score": round(float(scores[column]), 6),
                }
            )
        return out


def _describe(catalogue, stock_code: str) -> str | None:
    try:
        value = catalogue.get(stock_code)
    except Exception:  # noqa: BLE001 - a missing description is not an error
        return None
    return None if value is None else str(value)


def _bad_request(message: str):
    return jsonify({"error": message}), 400


def create_app(artifact_path: str | Path | None = None) -> Flask:
    """Build the application. ``artifact_path`` exists so tests can inject one."""
    if artifact_path is None:
        artifact_path = os.environ.get("RECENGINE_ARTIFACT", DEFAULT_ARTIFACT)

    app = Flask(__name__)
    store = ModelStore(Path(artifact_path))
    app.config["STORE"] = store

    @app.get("/")
    def index():
        return jsonify(
            {
                "service": "recengine",
                "version": __version__,
                "endpoints": {
                    "GET /health": "readiness and model metadata",
                    "GET /recommend": "top-N products for a customer; "
                    "parameters user_id (required), k (optional)",
                },
            }
        )

    @app.get("/health")
    def health():
        if not store.ready:
            return jsonify({"status": "no_model", "error": store.error}), 503
        return jsonify({"status": "ok", "version": __version__, **store.metadata})

    @app.get("/recommend")
    def recommend():
        if not store.ready:
            return jsonify({"status": "no_model", "error": store.error}), 503

        raw_user = request.args.get("user_id")
        if raw_user is None:
            return _bad_request("user_id parameter is required")
        try:
            user_id = int(raw_user)
        except ValueError:
            return _bad_request(f"user_id must be an integer, got {raw_user!r}")

        raw_k = request.args.get("k", DEFAULT_K)
        try:
            k = int(raw_k)
        except ValueError:
            return _bad_request(f"k must be an integer, got {raw_k!r}")
        if not 1 <= k <= MAX_K:
            return _bad_request(f"k must be between 1 and {MAX_K}, got {k}")

        try:
            recommendations = store.recommend(user_id, k)
        except KeyError:
            # 404 rather than the original's 400: the request was well formed,
            # the customer simply is not in the trained model.
            return (
                jsonify(
                    {
                        "error": f"unknown user_id {user_id}",
                        "hint": "this customer had no purchases in the training data",
                    }
                ),
                404,
            )

        return jsonify(
            {
                "user_id": user_id,
                "k": k,
                "model": store.metadata.get("model"),
                "recommendations": recommendations,
            }
        )

    return app


#: Module-level instance for `gunicorn recengine.api:app`.
app = create_app()
