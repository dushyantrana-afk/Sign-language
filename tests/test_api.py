import pytest

from backend.app.landmarks import NUM_LANDMARKS


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("SIGNBRIDGE_DATA_DIR", str(tmp_path))
    # Import after setting env var and reload module-level state so each
    # test gets an isolated data directory and a fresh in-memory classifier.
    from backend.app import main as main_module

    main_module._classifier = None
    main_module.app.config.update(TESTING=True)
    with main_module.app.test_client() as c:
        yield c


def fake_hand(variant="A", jitter=0.0, label="Right"):
    """One synthetic MediaPipe-shaped hand: 21 [x, y, z] points.

    `variant` picks a distinct relative hand *shape* (not just a global
    offset -- normalize_hand() is translation-invariant by design, so two
    hands that only differ by a shift would be indistinguishable after
    normalization, which is correct behavior but useless for this test).
    `jitter` nudges every coordinate slightly, to simulate "the same sign,
    captured again" without changing the shape.
    """
    sign = 1.0 if variant == "A" else -1.0
    return {
        "label": label,
        "landmarks": [
            [0.1 + 0.05 * i + jitter, sign * (0.2 + 0.07 * i) + jitter, 0.0]
            for i in range(NUM_LANDMARKS)
        ],
    }


def test_health(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.get_json() == {"status": "ok"}


def test_vocabulary_starts_empty(client):
    res = client.get("/api/vocabulary")
    data = res.get_json()
    assert data["labels"] == []
    assert data["ready_for_recognition"] is False


def test_enroll_rejects_missing_fields(client):
    res = client.post("/api/enroll", json={})
    assert res.status_code == 400


def test_enroll_with_no_hands_sent(client):
    res = client.post("/api/enroll", json={"label": "hello", "hands": []})
    assert res.status_code == 422
    assert res.get_json()["error"] == "no_hand_detected"


def test_enroll_rejects_malformed_landmarks(client):
    res = client.post(
        "/api/enroll",
        json={"label": "hello", "hands": [{"label": "Right", "landmarks": [[1, 2, 3]]}]},
    )
    assert res.status_code == 400


def test_enroll_then_vocabulary_reflects_it(client):
    res = client.post(
        "/api/enroll", json={"label": "hello", "hands": [fake_hand(variant="A")]}
    )
    assert res.status_code == 200
    data = res.get_json()
    assert data["label"] == "hello"
    assert data["examples_for_label"] == 1

    vocab = client.get("/api/vocabulary").get_json()
    assert vocab["labels"] == ["hello"]
    assert vocab["example_counts"]["hello"] == 1


def test_predict_with_no_hands_sent(client):
    res = client.post("/api/predict", json={"hands": []})
    assert res.status_code == 200
    data = res.get_json()
    assert data["reason"] == "no_hand_detected"
    assert data["needs_confirmation"] is True


def test_predict_requires_hands_field(client):
    res = client.post("/api/predict", json={})
    assert res.status_code == 400


def test_predict_before_training_says_not_trained(client):
    res = client.post("/api/predict", json={"hands": [fake_hand()]})
    assert res.status_code == 200
    data = res.get_json()
    assert data["reason"] == "not_trained"
    assert data["needs_confirmation"] is True


def test_enroll_two_labels_then_predict_recognizes_closest(client):
    for _ in range(3):
        client.post("/api/enroll", json={"label": "hello", "hands": [fake_hand(variant="A")]})
    for _ in range(3):
        client.post("/api/enroll", json={"label": "thank_you", "hands": [fake_hand(variant="B")]})

    status = client.get("/api/train_status").get_json()
    assert status["ready_for_recognition"] is True

    res = client.post("/api/predict", json={"hands": [fake_hand(variant="A", jitter=0.001)]})
    data = res.get_json()
    assert data["label"] == "hello"
    assert data["needs_confirmation"] is False
    assert data["reason"] == "ok"


def test_feedback_requires_consent(client):
    res = client.post(
        "/api/feedback",
        json={"predicted_label": "hello", "correct_label": "hello", "consent": False},
    )
    assert res.status_code == 200
    assert res.get_json()["stored"] is False


def test_feedback_with_consent_is_stored(client):
    res = client.post(
        "/api/feedback",
        json={"predicted_label": "hello", "correct_label": "hello", "consent": True},
    )
    assert res.status_code == 200
    assert res.get_json()["stored"] is True


def test_reset_clears_data(client):
    client.post("/api/enroll", json={"label": "hello", "hands": [fake_hand()]})
    res = client.post("/api/reset", json={})
    assert res.status_code == 200
    assert res.get_json()["reset"] is True
    vocab = client.get("/api/vocabulary").get_json()
    assert vocab["labels"] == []
