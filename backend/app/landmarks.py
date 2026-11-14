"""
Hand landmark normalization and feature-vector construction.

Where landmark *detection* happens: in the browser, via the MediaPipe
Hands JavaScript solution (frontend/app.js), not here. Two reasons:

1. This backend runs in a sandboxed environment whose installed
   `mediapipe` Python package ships neither the legacy `mediapipe.solutions`
   API nor the model-asset (`.task`) files the newer Tasks API needs, and
   this environment has no general internet access to download either —
   so server-side detection genuinely cannot run here. Confirmed by
   checking `mediapipe.solutions` (absent) and searching the filesystem
   for `*.task` model files (none present).
2. It is arguably the better architecture anyway: the browser already has
   the camera frame and a GPU-accelerated WASM hand model; shipping full
   video frames to a backend just to re-run detection server-side adds
   latency and a privacy cost (raw video leaving the device) for no
   benefit. Sending only the 21-point landmark coordinates per hand keeps
   video on-device, matching the "process on-device where possible" goal
   in /data-docs/consent.md.

So this module's job is: take the landmark JSON the frontend already
extracted, and turn it into the same fixed-length, translation- and
scale-invariant feature vector the classifier trains and predicts on.

Design notes (see /data-docs for the scope this supports):
- Supports up to two hands, since ISL (like most sign languages) uses
  two-handed signs.
- Each hand contributes 21 landmarks x 3 coords (x, y, z) = 63 numbers.
- Coordinates are made relative to the hand's own wrist and scaled by a
  stable reference distance (wrist -> middle-finger MCP) so the same sign
  produces a similar vector regardless of hand size or distance from the
  camera.
- If only one hand is visible, the second hand's slot is zero-filled and a
  `hand_count` feature records how many hands were actually seen, so the
  classifier can tell "no second hand" apart from "second hand at origin".

A server-side `HandLandmarkExtractor` (OpenCV + `mediapipe.solutions.hands`)
is still included below for deployments where a full MediaPipe install
*does* have `solutions` available (e.g. a normal developer machine, as
opposed to this sandbox) and a team prefers to keep raw frames server-side.
It is optional and unused by the default `/api/enroll` and `/api/predict`
routes, which take browser-extracted landmarks instead.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import mediapipe as mp
except ImportError:  # pragma: no cover - mediapipe should be installed
    mp = None

NUM_LANDMARKS = 21
COORDS_PER_LANDMARK = 3
PER_HAND_LENGTH = NUM_LANDMARKS * COORDS_PER_LANDMARK  # 63
FEATURE_LENGTH = PER_HAND_LENGTH * 2 + 1  # two hands + hand_count flag

WRIST_IDX = 0
MIDDLE_MCP_IDX = 9


@dataclass
class HandResult:
    """One detected hand's raw landmarks plus which side it is."""

    label: str  # "Left" or "Right" (as reported by MediaPipe, mirrored view)
    landmarks: np.ndarray  # shape (21, 3)


class HandLandmarkExtractor:
    """Thin, reusable wrapper around MediaPipe's static-image Hands model.

    Optional server-side path -- see the module docstring. Raises
    RuntimeError at construction time (not import time) if this
    environment's mediapipe build doesn't expose `solutions`, so importing
    this module never fails even where this class can't be used.
    """

    def __init__(self, max_num_hands: int = 2, min_detection_confidence: float = 0.5):
        if mp is None or not hasattr(mp, "solutions"):
            raise RuntimeError(
                "This mediapipe install has no 'solutions' API (common in "
                "minimal/offline builds). Use browser-side MediaPipe Hands "
                "JS and the hands_from_payload() path instead -- see this "
                "module's docstring."
            )
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
        )

    def close(self) -> None:
        self._hands.close()

    def __enter__(self) -> "HandLandmarkExtractor":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def extract(self, image_rgb: np.ndarray) -> list[HandResult]:
        """Run detection on an RGB image (H, W, 3) and return found hands."""
        result = self._hands.process(image_rgb)
        hands: list[HandResult] = []
        if not result.multi_hand_landmarks:
            return hands
        handedness_list = result.multi_handedness or []
        for i, hand_landmarks in enumerate(result.multi_hand_landmarks):
            coords = np.array(
                [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
                dtype=np.float64,
            )
            label = "Unknown"
            if i < len(handedness_list):
                label = handedness_list[i].classification[0].label
            hands.append(HandResult(label=label, landmarks=coords))
        return hands


def normalize_hand(landmarks: np.ndarray) -> np.ndarray:
    """Translate to wrist origin and scale by wrist->middle-MCP distance.

    Returns a flat (63,) vector. If the scale reference distance is ~0
    (degenerate/garbage input), returns a zero vector rather than dividing
    by a near-zero number.
    """
    wrist = landmarks[WRIST_IDX]
    centered = landmarks - wrist
    scale_ref = np.linalg.norm(centered[MIDDLE_MCP_IDX])
    if scale_ref < 1e-6:
        return np.zeros(PER_HAND_LENGTH, dtype=np.float64)
    normalized = centered / scale_ref
    return normalized.flatten()


def hands_to_feature_vector(hands: list[HandResult]) -> np.ndarray:
    """Combine up to two detected hands into one fixed-length feature vector.

    Ordering: "Left" hand goes in the first slot, "Right" in the second
    (as reported by MediaPipe), so the same physical sign lands in the same
    place in the vector whichever hand(s) performed it. Extra/unknown-label
    hands past the first two are ignored (two-handed signs are the ISL norm).
    """
    left = np.zeros(PER_HAND_LENGTH, dtype=np.float64)
    right = np.zeros(PER_HAND_LENGTH, dtype=np.float64)
    seen_left = False
    seen_right = False

    for hand in hands:
        vec = normalize_hand(hand.landmarks)
        if hand.label == "Left" and not seen_left:
            left = vec
            seen_left = True
        elif hand.label == "Right" and not seen_right:
            right = vec
            seen_right = True

    hand_count = float(len(hands))
    return np.concatenate([left, right, [hand_count]])


def feature_vector_from_rgb_image(
    image_rgb: np.ndarray, extractor: Optional[HandLandmarkExtractor] = None
) -> Optional[np.ndarray]:
    """Convenience: image -> feature vector, or None if no hand is detected.

    Server-side path (see module docstring); requires HandLandmarkExtractor
    to be constructible in this environment.
    """
    owns_extractor = extractor is None
    if extractor is None:
        extractor = HandLandmarkExtractor()
    try:
        hands = extractor.extract(image_rgb)
        if not hands:
            return None
        return hands_to_feature_vector(hands)
    finally:
        if owns_extractor:
            extractor.close()


class InvalidLandmarkPayload(ValueError):
    """Raised when the JSON a client sent doesn't look like MediaPipe Hands output."""


def hands_from_payload(raw_hands: object) -> list[HandResult]:
    """Parse the frontend's JSON landmark payload into HandResult objects.

    Expected shape (matches what MediaPipe Hands JS reports per frame, via
    `results.multiHandLandmarks` + `results.multiHandedness`):

        [
          {"label": "Left",  "landmarks": [[x, y, z], ... 21 points]},
          {"label": "Right", "landmarks": [[x, y, z], ... 21 points]}
        ]

    `label` is optional per entry (defaults to "Unknown"); `landmarks` must
    be exactly 21 [x, y, z] triples. Raises InvalidLandmarkPayload on any
    structural problem so the API layer can return a clean 400 instead of
    a raw exception.
    """
    if not isinstance(raw_hands, list):
        raise InvalidLandmarkPayload("expected a list of hands")
    if len(raw_hands) == 0:
        return []

    hands: list[HandResult] = []
    for i, entry in enumerate(raw_hands):
        if not isinstance(entry, dict) or "landmarks" not in entry:
            raise InvalidLandmarkPayload(f"hand[{i}] is missing 'landmarks'")
        points = entry["landmarks"]
        if not isinstance(points, list) or len(points) != NUM_LANDMARKS:
            raise InvalidLandmarkPayload(
                f"hand[{i}].landmarks must have exactly {NUM_LANDMARKS} points"
            )
        try:
            coords = np.array(
                [[float(p[0]), float(p[1]), float(p[2])] for p in points],
                dtype=np.float64,
            )
        except (TypeError, ValueError, IndexError) as exc:
            raise InvalidLandmarkPayload(f"hand[{i}].landmarks has non-numeric data") from exc
        label = str(entry.get("label", "Unknown"))
        hands.append(HandResult(label=label, landmarks=coords))
    return hands


def feature_vector_from_payload(raw_hands: object) -> Optional[np.ndarray]:
    """JSON landmark payload -> feature vector, or None if no hand was sent."""
    hands = hands_from_payload(raw_hands)
    if not hands:
        return None
    return hands_to_feature_vector(hands)
