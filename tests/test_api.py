"""Minimal automated tests for ML Simulator API using TestClient."""
from fastapi.testclient import TestClient
from api import app, _init_state  # type: ignore

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] in {"healthy", "online"}


def test_train_and_predict_flow():
    # Train
    train_payload = {"X": [[1], [2], [3], [4], [5]], "y": [2, 4, 5, 4, 5]}
    r = client.post("/api/v1/linear-regression/train", json=train_payload)
    assert r.status_code == 200, r.text
    train_data = r.json()
    assert train_data["model_type"] == "Linear Regression"
    # Predict
    predict_payload = {"features": [[6], [7]]}
    r = client.post("/api/v1/linear-regression/predict", json=predict_payload)
    assert r.status_code == 200, r.text
    pred = r.json()
    assert "predictions" in pred and len(pred["predictions"]) == 2


def test_train_and_predict_single_call():
    payload = {"X": [[1], [2], [3]], "y": [2, 4, 5]}
    r = client.post("/api/v1/linear-regression/train-and-predict", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert len(data["predictions"]) == 3


def test_predict_before_training():
    """Calling predict without prior training should return 400."""
    _init_state(app)  # reset model state explicitly
    fresh_client = TestClient(app)
    r = fresh_client.post(
        "/api/v1/linear-regression/predict",
        json={"features": [[1], [2]]},
    )
    assert r.status_code == 400, r.text
    assert "Model not trained" in r.text


def test_train_mismatched_lengths():
    """Training with mismatched X and y lengths should return 400."""
    _init_state(app)  # ensure clean state
    payload = {"X": [[1], [2], [3]], "y": [10, 20]}  # mismatch
    r = client.post("/api/v1/linear-regression/train", json=payload)
    assert r.status_code == 400, r.text
    assert "must have same length" in r.text


if __name__ == "__main__":  # Simple ad-hoc runner
    failures = 0
    for fn in [
        test_health,
        test_train_and_predict_flow,
        test_train_and_predict_single_call,
        test_predict_before_training,
        test_train_mismatched_lengths,
    ]:
        try:
            fn()
            print(f"[PASS] {fn.__name__}")
        except AssertionError as e:
            failures += 1
            print(f"[FAIL] {fn.__name__}: {e}")
    if failures:
        raise SystemExit(1)
    print("All tests passed.")
