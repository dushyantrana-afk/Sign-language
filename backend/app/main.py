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
2. Hand-landmark *detection* happens in the browser (MediaPipe Hands JS),
   not here -- see landmarks.py's module docstring for why. This backend
   receives already-extracted landmark coordinates, not raw images.

Scope reminder (see /data-docs): this recognizes a small, user-enrolled
vocabulary via hand landmarks. It is a gesture recognizer, not a general
sign-language translator, and it always offers a confirmation step instead
of silently trusting a low-confidence guess.
"""
from __future__ import annotations

import os

from flask import Flask, jsonify, request, send_from_directory

from . import storage
from .classifier import SignKNNClassifier
from .landmarks import InvalidLandmarkPayload, feature_vector_from_payload

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "frontend")

app = Flask(__name__, static_folder=None)

_classifier: SignKNNClassifier | None = None


def get_classifier() -> SignKNNClassifier:
    global _classifier
    if _classifier is None:
        state = storage.load_classifier_state()
        _classifier = SignKNNClassifier.from_state(state)
    return _classifier


def persist_classifier() -> None:
    storage.save_classifier_state(get_classifier().to_state())


@app.get("/api/health")
def health():
    return jsonify({"status": "ok"})


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

    {"label": "hello", "hands": [{"label": "Right", "landmarks": [[x,y,z], ...21]}]}

    `hands` is the frontend's MediaPipe Hands JS output for one captured
    frame -- see landmarks.hands_from_payload() for the exact shape.
    """
    payload = request.get_json(silent=True) or {}
    label = (payload.get("label") or "").strip()
    raw_hands = payload.get("hands")

    if not label:
        return jsonify({"error": "label is required"}), 400
    if raw_hands is None:
        return jsonify({"error": "hands is required"}), 400

    try:
        vector = feature_vector_from_payload(raw_hands)
    except InvalidLandmarkPayload as exc:
        return jsonify({"error": f"invalid hands payload: {exc}"}), 400

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
            "hands_detected": len(raw_hands),
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
    """Predict the sign for one frame's landmarks.

    {"hands": [{"label": "Right", "landmarks": [[x,y,z], ...21]}]}

    Always returns a confidence and a needs_confirmation flag -- the
    frontend must show a confirm/correct step whenever that flag is true,
    per the deck's "never silently convert uncertain predictions" rule.
    """
    payload = request.get_json(silent=True) or {}
    raw_hands = payload.get("hands")
    if raw_hands is None:
        return jsonify({"error": "hands is required"}), 400

    try:
        vector = feature_vector_from_payload(raw_hands)
    except InvalidLandmarkPayload as exc:
        return jsonify({"error": f"invalid hands payload: {exc}"}), 400

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
            "hands_detected": len(raw_hands),
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
