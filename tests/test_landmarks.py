import numpy as np
import pytest

from backend.app.landmarks import (
    FEATURE_LENGTH,
    NUM_LANDMARKS,
    PER_HAND_LENGTH,
    HandResult,
    InvalidLandmarkPayload,
    feature_vector_from_payload,
    hands_from_payload,
    hands_to_feature_vector,
    normalize_hand,
)


def make_landmarks(wrist=(0, 0, 0), middle_mcp=(0, 1, 0), fill=0.1):
    """21x3 synthetic landmarks with a controllable wrist and scale reference."""
    coords = np.full((21, 3), fill, dtype=np.float64)
    coords[0] = wrist
    coords[9] = middle_mcp
    return coords


def test_normalize_hand_is_translation_invariant():
    base = make_landmarks(wrist=(0, 0, 0), middle_mcp=(0, 1, 0))
    shifted = make_landmarks(wrist=(5, 5, 5), middle_mcp=(5, 6, 5))
    # shift every point by the same offset -> same relative geometry
    shifted = shifted.copy()
    shifted[:, :] += 0  # already constructed with offset wrist/mcp; others at fixed fill
    # Build a properly shifted version by adding (5,5,5) to the base array instead.
    shifted2 = base + np.array([5, 5, 5])
    n1 = normalize_hand(base)
    n2 = normalize_hand(shifted2)
    assert np.allclose(n1, n2, atol=1e-8)


def test_normalize_hand_is_scale_invariant():
    base = make_landmarks(wrist=(0, 0, 0), middle_mcp=(0, 1, 0), fill=0.2)
    scaled = base * 3.0
    n1 = normalize_hand(base)
    n2 = normalize_hand(scaled)
    assert np.allclose(n1, n2, atol=1e-6)


def test_normalize_hand_degenerate_scale_returns_zeros():
    degenerate = np.zeros((21, 3))  # wrist == middle_mcp -> zero scale ref
    result = normalize_hand(degenerate)
    assert result.shape == (PER_HAND_LENGTH,)
    assert np.allclose(result, 0.0)


def test_hands_to_feature_vector_shape_and_hand_count():
    left = HandResult(label="Left", landmarks=make_landmarks())
    right = HandResult(label="Right", landmarks=make_landmarks())

    one_hand_vec = hands_to_feature_vector([left])
    two_hand_vec = hands_to_feature_vector([left, right])

    assert one_hand_vec.shape == (FEATURE_LENGTH,)
    assert two_hand_vec.shape == (FEATURE_LENGTH,)
    assert one_hand_vec[-1] == 1.0
    assert two_hand_vec[-1] == 2.0
    # right-hand slot should be all zeros when only the left hand is seen
    right_slot = one_hand_vec[PER_HAND_LENGTH : PER_HAND_LENGTH * 2]
    assert np.allclose(right_slot, 0.0)


def test_hands_to_feature_vector_empty_input():
    vec = hands_to_feature_vector([])
    assert vec.shape == (FEATURE_LENGTH,)
    assert vec[-1] == 0.0


# ---- hands_from_payload / feature_vector_from_payload (frontend JSON path) ----


def make_payload_hand(label="Right"):
    return {"label": label, "landmarks": [[0.1 * i, 0.2 * i, 0.0] for i in range(NUM_LANDMARKS)]}


def test_hands_from_payload_parses_valid_input():
    raw = [make_payload_hand("Left"), make_payload_hand("Right")]
    hands = hands_from_payload(raw)
    assert len(hands) == 2
    assert hands[0].label == "Left"
    assert hands[0].landmarks.shape == (NUM_LANDMARKS, 3)


def test_hands_from_payload_empty_list_is_fine():
    assert hands_from_payload([]) == []


def test_hands_from_payload_rejects_non_list():
    with pytest.raises(InvalidLandmarkPayload):
        hands_from_payload({"not": "a list"})


def test_hands_from_payload_rejects_wrong_point_count():
    bad = [{"label": "Right", "landmarks": [[0, 0, 0]]}]
    with pytest.raises(InvalidLandmarkPayload):
        hands_from_payload(bad)


def test_hands_from_payload_rejects_missing_landmarks_key():
    with pytest.raises(InvalidLandmarkPayload):
        hands_from_payload([{"label": "Right"}])


def test_hands_from_payload_defaults_label_to_unknown():
    raw = [{"landmarks": [[0.0, 0.0, 0.0]] * NUM_LANDMARKS}]
    hands = hands_from_payload(raw)
    assert hands[0].label == "Unknown"


def test_feature_vector_from_payload_none_when_empty():
    assert feature_vector_from_payload([]) is None


def test_feature_vector_from_payload_matches_manual_path():
    raw = [make_payload_hand("Left")]
    vec = feature_vector_from_payload(raw)
    expected = hands_to_feature_vector(hands_from_payload(raw))
    assert np.allclose(vec, expected)
