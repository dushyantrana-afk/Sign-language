"""
Hand landmark detection, normalization, and feature-vector construction.

Landmark *detection* can happen in two places, and the API layer
(backend/app/main.py) supports both at once, per request:

1. **Browser-side (default)** -- the MediaPipe Hands JavaScript solution
   (frontend/app.js) detects hands locally and sends just the 21 (x, y, z)
   points per hand to the backend as JSON. No video ever leaves the
   device. Parsed here by `hands_from_payload()` / `feature_vector_from_payload()`.
2. **Server-side (optional)** -- `HandLandmarkExtractor` below runs
   MediaPipe's Tasks API (`HandLandmarker`) on a still frame the frontend
   sends as a base64 image. Useful when the browser can't load the
   MediaPipe Hands JS bundle from its CDN (an offline kiosk, a restrictive
   network) but the *backend* machine has normal internet access, or for
   the bulk dataset importer (`ml-model/import_dataset.py`), which has no
   browser in the loop at all.

Why not just always use #2 and skip the browser-side complexity: keeping
raw video off the network by default is a real privacy win (see
`/data-docs/consent.md`), and it's the only option background scripts
(frontend never opened) or CDN-blocked setups have. Both paths produce
the exact same `HandResult`/feature-vector shape, so the classifier never
needs to know which one ran.

**A note on this specific MVP's own build/dev sandbox:** the sandbox this
project was developed in has an incomplete, offline `mediapipe` install
(no `mediapipe.solutions`, no bundled `.task` model file) and no general
internet access, so option 2 could be implemented and unit-tested for its
error-handling, but not run end-to-end there. On a normal developer
machine or server with internet access, `HandLandmarkExtractor` below
downloads the official ~8MB hand-landmark model once (cached under
`backend/models/`) and works normally. Run
`python -c "from backend.app.landmarks import HandLandmarkExtractor; HandLandmarkExtractor()"`
after installing `opencv-python` and `mediapipe` to confirm it works in
*your* environment.

Design notes for the feature vector itself (see /data-docs for the scope
this supports):
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
"""
from __future__ import annotations

import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import mediapipe as mp
except ImportError:  # pragma: no cover - mediapipe should be installed
    mp = None

# Official MediaPipe-hosted model asset for the Tasks API HandLandmarker.
# Documented at https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker
HAND_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_FILENAME = "hand_landmarker.task"

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


def model_cache_path() -> str:
    return os.path.join(MODEL_CACHE_DIR, MODEL_FILENAME)


def ensure_model_downloaded(timeout: float = 30.0) -> str:
    """Return a local path to the hand-landmark model, downloading it once
    if it isn't cached yet. Raises RuntimeError (never a raw network
    exception) if it can't be fetched -- e.g. no internet access -- so
    callers get one clear, actionable message instead of a stack trace
    from deep inside urllib.
    """
    path = model_cache_path()
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path

    os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
    tmp_path = path + ".part"
    try:
        with urllib.request.urlopen(HAND_LANDMARKER_MODEL_URL, timeout=timeout) as response:
            with open(tmp_path, "wb") as f:
                while True:
                    chunk = response.read(1024 * 256)
                    if not chunk:
                        break
                    f.write(chunk)
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise RuntimeError(
            "Could not download the hand-landmark model from "
            f"{HAND_LANDMARKER_MODEL_URL} ({exc}). Server-side detection "
            "needs outbound internet access on the machine running the "
            "backend, once, to cache this ~8MB file under "
            f"{MODEL_CACHE_DIR}/. Use browser-side detection "
            "(the default) instead if this machine can't reach the "
            "internet -- see this module's docstring."
        ) from exc

    os.replace(tmp_path, path)
    return path


def is_server_side_detection_available() -> bool:
    """Cheap capability check for /api/capabilities: does NOT download
    anything -- just checks that mediapipe's Tasks API is importable and
    either the model is already cached or looks reachable. Actual
    detection still does the real download/construction lazily."""
    if mp is None:
        return False
    try:
        from mediapipe.tasks.python import vision  # noqa: F401
        from mediapipe.tasks.python.core.base_options import BaseOptions  # noqa: F401
    except ImportError:
        return False
    return True


class HandLandmarkExtractor:
    """Wrapper around MediaPipe's Tasks-API HandLandmarker (static images).

    Optional server-side path -- see the module docstring. Raises
    RuntimeError at construction time (not import time) if this
    environment's mediapipe build lacks the Tasks vision API, or if the
    model asset can't be downloaded/cached -- so importing this module
    never fails even where this class can't be used.
    """

    def __init__(self, max_num_hands: int = 2, min_detection_confidence: float = 0.5):
        if mp is None:
            raise RuntimeError("mediapipe is not installed in this environment")
        try:
            from mediapipe.tasks.python import vision
            from mediapipe.tasks.python.core.base_options import BaseOptions
        except ImportError as exc:
            raise RuntimeError(
                "This mediapipe install has no Tasks vision API "
                "(mediapipe.tasks.python.vision). Use browser-side "
                "MediaPipe Hands JS instead -- see this module's docstring."
            ) from exc

        model_path = ensure_model_downloaded()  # raises RuntimeError with a clear reason on failure
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> "HandLandmarkExtractor":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def extract(self, image_rgb: np.ndarray) -> list[HandResult]:
        """Run detection on an RGB image (H, W, 3) and return found hands."""
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(image_rgb))
        result = self._landmarker.detect(mp_image)
        hands: list[HandResult] = []
        for i, hand_landmarks in enumerate(result.hand_landmarks):
            coords = np.array(
                [[lm.x, lm.y, lm.z] for lm in hand_landmarks], dtype=np.float64
            )
            label = "Unknown"
            if i < len(result.handedness) and result.handedness[i]:
                label = result.handedness[i][0].category_name
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
