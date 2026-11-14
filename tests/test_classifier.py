import numpy as np
import pytest

from backend.app.classifier import MIN_EXAMPLES_PER_LABEL, SignKNNClassifier


def make_cluster(center: np.ndarray, n: int, noise: float = 0.01, seed: int = 0):
    rng = np.random.default_rng(seed)
    return [center + rng.normal(scale=noise, size=center.shape) for _ in range(n)]


def test_not_trainable_with_fewer_than_two_labels():
    clf = SignKNNClassifier()
    center = np.ones(10)
    for vec in make_cluster(center, MIN_EXAMPLES_PER_LABEL):
        clf.add_example(vec, "hello")
    assert not clf.is_trainable()
    result = clf.predict(center)
    assert result.reason == "not_trained"
    assert result.needs_confirmation is True


def test_predicts_confident_label_for_well_separated_clusters():
    clf = SignKNNClassifier(confidence_threshold=0.6)
    hello_center = np.zeros(10)
    thanks_center = np.ones(10) * 10  # far away -> easy to separate

    for vec in make_cluster(hello_center, 5, seed=1):
        clf.add_example(vec, "hello")
    for vec in make_cluster(thanks_center, 5, seed=2):
        clf.add_example(vec, "thank_you")

    assert clf.is_trainable()

    result = clf.predict(hello_center)
    assert result.label == "hello"
    assert result.needs_confirmation is False
    assert result.reason == "ok"
    assert result.confidence >= 0.6


def test_far_away_vector_is_no_match_not_a_forced_guess():
    clf = SignKNNClassifier()
    hello_center = np.zeros(10)
    thanks_center = np.ones(10) * 10

    for vec in make_cluster(hello_center, 5, seed=1):
        clf.add_example(vec, "hello")
    for vec in make_cluster(thanks_center, 5, seed=2):
        clf.add_example(vec, "thank_you")

    far_away = np.ones(10) * -50
    result = clf.predict(far_away)
    assert result.reason == "no_match"
    assert result.label is None
    assert result.needs_confirmation is True


def test_state_roundtrip_preserves_predictions():
    clf = SignKNNClassifier()
    hello_center = np.zeros(10)
    thanks_center = np.ones(10) * 10
    for vec in make_cluster(hello_center, 5, seed=1):
        clf.add_example(vec, "hello")
    for vec in make_cluster(thanks_center, 5, seed=2):
        clf.add_example(vec, "thank_you")

    state = clf.to_state()
    restored = SignKNNClassifier.from_state(state)

    result = restored.predict(hello_center)
    assert result.label == "hello"
    assert result.reason == "ok"


def test_label_counts():
    clf = SignKNNClassifier()
    clf.add_example(np.zeros(4), "a")
    clf.add_example(np.zeros(4), "a")
    clf.add_example(np.zeros(4), "b")
    assert clf.label_counts() == {"a": 2, "b": 1}
