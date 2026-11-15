"""
Tests for the two-tier storage split: a shipped, git-committed `library.json`
(built offline from a licensed dataset) merged with a user's own live
enrollments in `classifier_state.json` (gitignored, personal data).
See backend/app/storage.py's module docstring for the full reasoning.
"""
import json
import os

import pytest

from backend.app.landmarks import NUM_LANDMARKS


def fake_hand(variant="A", jitter=0.0, label="Right"):
    sign = 1.0 if variant == "A" else -1.0
    return {
        "label": label,
        "landmarks": [
            [0.1 + 0.05 * i + jitter, sign * (0.2 + 0.07 * i) + jitter, 0.0]
            for i in range(NUM_LANDMARKS)
        ],
    }


def write_library(data_dir, labels_and_variants):
    """labels_and_variants: list of (label, variant) -- writes 3 examples
    each straight into library.json, bypassing import_dataset.py (which
    needs real mediapipe detection, unavailable in this sandbox)."""
    vectors, labels = [], []
    from backend.app.landmarks import feature_vector_from_payload

    for label, variant in labels_and_variants:
        for _ in range(3):
            vec = feature_vector_from_payload([fake_hand(variant=variant)])
            vectors.append(vec.tolist())
            labels.append(label)
    path = os.path.join(data_dir, "library.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"vectors": vectors, "labels": labels}, f)


@pytest.fixture()
def fresh_main(tmp_path, monkeypatch):
    monkeypatch.setenv("SIGNBRIDGE_DATA_DIR", str(tmp_path))
    from backend.app import main as main_module

    main_module._classifier = None
    main_module._user_vectors = []
    main_module._user_labels = []
    main_module.app.config.update(TESTING=True)
    return main_module, tmp_path


def test_vocabulary_with_no_library_is_empty(fresh_main):
    main_module, _ = fresh_main
    with main_module.app.test_client() as c:
        data = c.get("/api/vocabulary").get_json()
        assert data["labels"] == []
        assert data["library_labels"] == []


def test_library_signs_are_available_without_enrollment(fresh_main):
    main_module, data_dir = fresh_main
    write_library(str(data_dir), [("hello", "A"), ("thank_you", "B")])
    main_module._classifier = None  # force reload from the library file we just wrote

    with main_module.app.test_client() as c:
        vocab = c.get("/api/vocabulary").get_json()
        assert vocab["labels"] == ["hello", "thank_you"]
        assert vocab["library_labels"] == ["hello", "thank_you"]
        assert vocab["ready_for_recognition"] is True

        res = c.post("/api/predict", json={"hands": [fake_hand(variant="A", jitter=0.001)]})
        data = res.get_json()
        assert data["label"] == "hello"
        assert data["reason"] == "ok"


def test_user_enrollment_adds_to_library_without_modifying_it(fresh_main):
    main_module, data_dir = fresh_main
    write_library(str(data_dir), [("hello", "A"), ("thank_you", "B")])
    main_module._classifier = None

    with main_module.app.test_client() as c:
        for _ in range(3):
            c.post("/api/enroll", json={"label": "please", "hands": [fake_hand(variant="A", jitter=9.0)]})

        vocab = c.get("/api/vocabulary").get_json()
        assert set(vocab["labels"]) == {"hello", "thank_you", "please"}
        assert vocab["library_labels"] == ["hello", "thank_you"]  # library itself unchanged

    # The library file on disk must be untouched by enrollment.
    library_path = os.path.join(str(data_dir), "library.json")
    with open(library_path) as f:
        library_data = json.load(f)
    assert set(library_data["labels"]) == {"hello", "thank_you"}

    # But the user's own file now has the new example.
    user_path = os.path.join(str(data_dir), "classifier_state.json")
    with open(user_path) as f:
        user_data = json.load(f)
    assert user_data["labels"] == ["please", "please", "please"]


def test_reset_clears_user_layer_but_preserves_library(fresh_main):
    main_module, data_dir = fresh_main
    write_library(str(data_dir), [("hello", "A"), ("thank_you", "B")])
    main_module._classifier = None

    with main_module.app.test_client() as c:
        for _ in range(3):
            c.post("/api/enroll", json={"label": "please", "hands": [fake_hand(variant="A", jitter=9.0)]})

        res = c.post("/api/reset", json={})
        assert res.get_json() == {"reset": True, "library_preserved": True}

        vocab = c.get("/api/vocabulary").get_json()
        assert vocab["labels"] == ["hello", "thank_you"]  # "please" (user) is gone
        assert vocab["library_labels"] == ["hello", "thank_you"]  # library intact

    # classifier_state.json should be gone/empty; library.json still there.
    assert not os.path.exists(os.path.join(str(data_dir), "classifier_state.json"))
    assert os.path.exists(os.path.join(str(data_dir), "library.json"))
