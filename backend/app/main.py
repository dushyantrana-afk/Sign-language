"""
SignBridge AI backend (hackathon MVP).

Matches Module 3 ("API & Interface") from the pitch deck: a prediction
endpoint fed by camera input, plus enrollment, vocabulary, and consented
feedback endpoints. Two deliberate deviations from the deck's exact tech
list, both explained in code comments at the point they matter:

1. Flask instead of FastAPI -- this sandbox has no outbound package-install
   access, and Flask (unlike FastAPI) was already present. All the
   sign-recognition logic lives in landmarks.py/classifier.py, not here,
   so swapping web frameworks later is a routing-layer change only.
2. Hand-landmark detection defaults to the browser (MediaPipe Hands JS),
   with an optional server-side fallback -- see landmarks.py's module
   docstring. `/api/enroll` and `/api/predict` accept EITHER a "hands"
   field (landmarks already extracted client-side -- preferred) OR an
   "image" field (a base64 frame, detected here via HandLandmarkExtractor).
   If both are sent, "hands" wins.

Scope reminder (see /data-docs): this recognizes a small, user-enrolled
vocabulary via hand landmarks. It is a gesture recognizer, not a general
sign-language translator, and it always offers a confirmation step instead
of silently trusting a low-confidence guess.
"""
from __future__ import annotations

import base64
import binascii
import os

import numpy as np
from flask import Flask, jsonify, request, send_from_directory

from . import storage
from .classifier import SignKNNClassifier
from .landmarks import (
    HandLandmarkExtractor,
    InvalidLandmarkPayload,
    feature_vector_from_payload,
    is_server_side_detection_available,
)

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "frontend")

app = Flask(__name__, static_folder=None)

_classifier: SignKNNClassifier | None = None
_server_extractor: HandLandmarkExtractor | None = None
_server_extractor_error: str | None = None


def get_classifier() -> SignKNNClassifier:
    global _classifier
    if _classifier is None:
        state = storage.load_classifier_state()
        _classifier = SignKNNClassifier.from_state(state)
    return _classifier


def persist_classifier() -> None:
    storage.save_classifier_state(get_classifier().to_state())


def get_server_extractor() -> HandLandmarkExtractor:
    """Lazily construct the optional server-side extractor.

    Lazy (not at import/module load time) so the whole app still starts
    fine on a machine where this fails -- browser-side detection keeps
    working either way. The failure reason is cached so a backend with no
    internet access doesn't retry (and re-fail slowly) on every request.
    """
    global _server_extractor, _server_extractor_error
    if _server_extractor is not None:
        return _server_extractor
    if _server_extractor_error is not None:
        raise RuntimeError(_server_extractor_error)
    try:
        _server_extractor = HandLandmarkExtractor(max_num_hands=2, min_detection_confidence=0.5)
        return _server_extractor
    except RuntimeError as exc:
        _server_extractor_error = str(exc)
        raise


def decode_data_url_image(data_url: str) -> np.ndarray:
    """Decode a base64 data URL (e.g. 'data:image/jpeg;base64,...') to an
    RGB numpy image, as produced by a browser <canvas>.toDataURL() capture."""
    import cv2  # imported lazily: only the server-side image path needs it

    if "," in data_url:
        _, b64data = data_url.split(",", 1)
    else:
        b64data = data_url
    try:
        raw = base64.b64decode(b64data)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("invalid base64 image data") from exc
    arr = np.frombuffer(raw, dtype=np.uint8)
    image_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError("could not decode image data")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def feature_vector_from_request(payload: dict) -> tuple[np.ndarray | None, int, str | None]:
    """Resolve a feature vector from a request body that may carry "hands"
    (preferred, browser-extracted landmarks) or "image" (base64 frame,
    detected server-side). Returns (vector_or_None, hands_detected_count,
    error_or_None). `error` is set only for a hard failure (bad payload,
    server-side detection unavailable) -- "no hand visible" is reported as
    (None, 0, None), not an error.
    """
    raw_hands = payload.get("hands")
    image_b64 = payload.get("image")

    if raw_hands is not None:
        try:
            vector = feature_vector_from_payload(raw_hands)
        except InvalidLandmarkPayload as exc:
            return None, 0, f"invalid hands payload: {exc}"
        return vector, len(raw_hands), None

    if image_b64 is not None:
        try:
            extractor = get_server_extractor()
        except RuntimeError as exc:
            return None, 0, f"server_side_detection_unavailable: {exc}"
        try:
            image_rgb = decode_data_url_image(image_b64)
        except ValueError as exc:
            return None, 0, str(exc)
        hands = extractor.extract(image_rgb)
        if not hands:
            return None, 0, None
        from .landmarks import hands_to_feature_vector

        return hands_to_feature_vector(hands), len(hands), None

    return None, 0, "either 'hands' or 'image' is required"


@app.get("/api/health")
def health():
    return jsonify({"status": "ok"})


@app.get("/api/capabilities")
def capabilities():
    """Lets the frontend know whether it can offer a 'server-side
    detection' option at all, without triggering a model download just to
    check (is_server_side_detection_available() only checks that the
    Tasks API is importable, not that the model is reachable -- a real
    attempt can still fail with no internet, which /api/enroll and
    /api/predict report per-request via server_side_detection_unavailable).
    """
    return jsonify(
        {
            "server_side_detection_importable": is_server_side_detection_available(),
            "server_side_detection_ready": _server_extractor is not None,
        }
    )


@app.get("/api/vocabulary")
def get_vocabulary():
    clf = get_classifier()
    counts = clf.label_counts()
    return jsonify(
        {
            "labels": sorted(counts.keys()),
            "example_counts": counts,
            "ready_for_recognition": clf.is_trainable(),
        }
    )


@app.post("/api/enroll")
def enroll():
    """Record one example of a sign.

    Either (preferred, browser-side detection):
      {"label": "hello", "hands": [{"label": "Right", "landmarks": [[x,y,z], ...21]}]}
    or (optional, server-side detection -- see landmarks.py docstring):
      {"label": "hello", "image": "data:image/jpeg;base64,..."}

    `hands` is the frontend's MediaPipe Hands JS output for one captured
    frame -- see landmarks.hands_from_payload() for the exact shape.
    """
    payload = request.get_json(silent=True) or {}
    label = (payload.get("label") or "").strip()

    if not label:
        return jsonify({"error": "label is required"}), 400

    vector, hand_count, error = feature_vector_from_request(payload)
    if error is not None:
        status = 503 if error.startswith("server_side_detection_unavailable") else 400
        return jsonify({"error": error}), status
    if vector is None:
        return jsonify({"error": "no_hand_detected"}), 422

    clf = get_classifier()
    clf.add_example(vector, label)
    clf.fit()
    persist_classifier()

    counts = clf.label_counts()
    return jsonify(
        {
            "label": label,
            "hands_detected": hand_count,
            "examples_for_label": counts.get(label, 0),
            "ready_for_recognition": clf.is_trainable(),
        }
    )


@app.get("/api/train_status")
def train_status():
    clf = get_classifier()
    return jsonify(
        {
            "label_counts": clf.label_counts(),
            "ready_for_recognition": clf.is_trainable(),
            "min_examples_per_label": 3,
        }
    )


@app.post("/api/predict")
def predict():
    """Predict the sign for one frame.

    Either (preferred, browser-side detection):
      {"hands": [{"label": "Right", "landmarks": [[x,y,z], ...21]}]}
    or (optional, server-side detection -- see landmarks.py docstring):
      {"image": "data:image/jpeg;base64,..."}

    Always returns a confidence and a needs_confirmation flag -- the
    frontend must show a confirm/correct step whenever that flag is true,
    per the deck's "never silently convert uncertain predictions" rule.
    """
    payload = request.get_json(silent=True) or {}

    vector, hand_count, error = feature_vector_from_request(payload)
    if error is not None:
        status = 503 if error.startswith("server_side_detection_unavailable") else 400
        return jsonify({"error": error}), status

    if vector is None:
        return jsonify(
            {
                "label": None,
                "confidence": 0.0,
                "needs_confirmation": True,
                "reason": "no_hand_detected",
                "hands_detected": 0,
            }
        )

    clf = get_classifier()
    result = clf.predict(vector)
    return jsonify(
        {
            "label": result.label,
            "confidence": round(result.confidence, 3),
            "needs_confirmation": result.needs_confirmation,
            "reason": result.reason,
            "hands_detected": hand_count,
        }
    )


@app.post("/api/feedback")
def feedback():
    """Store a consented correction/feedback entry.

    {"predicted_label": "...", "correct_label": "...", "consent": true}

    Entries are only stored when consent is explicitly true, matching the
    deck's "store feedback only with consent" principle.
    """
    payload = request.get_json(silent=True) or {}
    if not payload.get("consent"):
        return jsonify({"stored": False, "reason": "consent_required"}), 200

    entry = {
        "predicted_label": payload.get("predicted_label"),
        "correct_label": payload.get("correct_label"),
        "confidence": payload.get("confidence"),
        "note": payload.get("note", ""),
    }
    storage.append_feedback(entry)
    return jsonify({"stored": True})


@app.post("/api/reset")
def reset():
    """Delete all enrolled signs and feedback (data minimization / demo reset)."""
    global _classifier
    storage.reset_all()
    _classifier = None
    return jsonify({"reset": True})


# ---- Static frontend (single-command demo: `python -m backend.app.main`) ----


@app.get("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.get("/<path:filename>")
def frontend_assets(filename):
    return send_from_directory(FRONTEND_DIR, filename)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True, use_reloader=False)
