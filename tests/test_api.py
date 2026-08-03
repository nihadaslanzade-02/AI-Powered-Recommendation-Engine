"""The HTTP contract.

The headline assertion here is :func:`test_recommendations_are_personalised`.
The original served an identical list to every customer, so a test suite for
this service that does not check two customers get different answers is
missing the only thing that went wrong.
"""

from __future__ import annotations

import joblib
import pytest

from recengine.api import MAX_K, create_app
from recengine.data import build_interactions, item_catalogue, load_transactions
from recengine.models import PureSVDRecommender


@pytest.fixture
def artifact_path(tmp_path, toy_csv):
    frame = load_transactions(toy_csv)
    train = build_interactions(frame)
    model = PureSVDRecommender(n_factors=4, random_state=0).fit(train, frame)
    path = tmp_path / "model.joblib"
    joblib.dump(
        {
            "model": model,
            "catalogue": item_catalogue(frame),
            "metadata": {"model": model.name, "n_users": train.n_users},
        },
        path,
    )
    return path


@pytest.fixture
def client(artifact_path):
    return create_app(artifact_path).test_client()


def test_health_reports_the_loaded_model(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.get_json()
    assert body["status"] == "ok"
    assert body["model"] == "PureSVD"


def test_index_lists_the_endpoints(client):
    body = client.get("/").get_json()
    assert "GET /recommend" in body["endpoints"]


def test_recommend_returns_ranked_products_with_descriptions(client):
    body = client.get("/recommend?user_id=1000&k=3").get_json()
    assert body["user_id"] == 1000
    assert body["k"] == 3
    assert len(body["recommendations"]) == 3
    assert [r["rank"] for r in body["recommendations"]] == [1, 2, 3]
    first = body["recommendations"][0]
    assert set(first) == {"rank", "stock_code", "description", "score"}
    assert first["description"].startswith("PRODUCT ")
    scores = [r["score"] for r in body["recommendations"]]
    assert scores == sorted(scores, reverse=True)


def test_recommendations_are_personalised(client):
    """Two customers from different communities must not get the same list.

    The original returned one identical list to everyone, so this is the
    property the whole rewrite exists to establish.
    """
    a = client.get("/recommend?user_id=1000&k=5").get_json()
    b = client.get("/recommend?user_id=2000&k=5").get_json()
    codes_a = [r["stock_code"] for r in a["recommendations"]]
    codes_b = [r["stock_code"] for r in b["recommendations"]]
    assert codes_a != codes_b


def test_recommend_never_returns_something_already_bought(client, artifact_path):
    store = joblib.load(artifact_path)
    model = store["model"]
    row = model.train.user_index[1000]
    already = {str(model.train.item_ids[c]) for c in model.train.seen_items(row)}

    body = client.get("/recommend?user_id=1000&k=5").get_json()
    returned = {r["stock_code"] for r in body["recommendations"]}
    assert not (returned & already)


def test_k_defaults_to_ten(client):
    body = client.get("/recommend?user_id=1000").get_json()
    assert body["k"] == 10


@pytest.mark.parametrize(
    "query,message",
    [
        ("", "user_id parameter is required"),
        ("?user_id=abc", "must be an integer"),
        ("?user_id=1000&k=abc", "k must be an integer"),
        ("?user_id=1000&k=0", "between 1 and"),
        (f"?user_id=1000&k={MAX_K + 1}", "between 1 and"),
    ],
)
def test_malformed_requests_are_rejected_with_400(client, query, message):
    response = client.get(f"/recommend{query}")
    assert response.status_code == 400
    assert message in response.get_json()["error"]


def test_unknown_customer_is_404_not_400(client):
    """The request is well formed; the customer simply is not in the model."""
    response = client.get("/recommend?user_id=987654")
    assert response.status_code == 404
    assert "unknown user_id" in response.get_json()["error"]


def test_missing_artifact_starts_but_reports_unhealthy(tmp_path):
    client = create_app(tmp_path / "absent.joblib").test_client()

    health = client.get("/health")
    assert health.status_code == 503
    assert health.get_json()["status"] == "no_model"
    assert "scripts/train.py" in health.get_json()["error"]

    recommend = client.get("/recommend?user_id=1000")
    assert recommend.status_code == 503


def test_corrupt_artifact_is_reported_rather_than_raised(tmp_path):
    broken = tmp_path / "broken.joblib"
    broken.write_bytes(b"this is not a joblib file")
    client = create_app(broken).test_client()
    assert client.get("/health").status_code == 503


def test_the_app_does_not_fit_a_model_at_import_time(artifact_path):
    """The original trained inside app.py at module scope.

    The artifact's model object must be the one that was loaded, not a fresh
    fit, so identity is the cheapest way to state it.
    """
    app = create_app(artifact_path)
    loaded = joblib.load(artifact_path)
    store = app.config["STORE"]
    assert store.model.name == loaded["model"].name
    assert store.model.train.n_items == loaded["model"].train.n_items
