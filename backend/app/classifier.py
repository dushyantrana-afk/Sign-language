"""
Sign classifier: a small, honest nearest-neighbor model over hand-landmark
feature vectors, with an explicit "I'm not confident" outcome instead of
always forcing a guess.

Why KNN instead of a deep model for the MVP: the pitch deck scopes this
release to "a small verified vocabulary" collected by live enrollment
(record a few examples per sign), not a pretrained dataset. KNN over
normalized landmark vectors works well for that regime and needs no
training data beyond what a user enrolls in the app. `/ml-model` documents
the upgrade path to a trained sequence model once real, consented data
exists.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

MIN_EXAMPLES_PER_LABEL = 3
# Euclidean distance (in normalized-landmark-vector space, see landmarks.py)
# beyond this to the nearest neighbor -> treat as "unknown gesture", not a
# forced guess at the closest enrolled label. Euclidean rather than cosine:
# feature vectors legitimately contain all-zero segments (the unused-hand
# slot for a one-handed sign), and cosine distance is undefined for a
# zero vector. This threshold is a starting heuristic, not calibrated
# against real enrollment data -- see /ml-model/README.md for how to tune
# it once real usage data exists.
MAX_NEIGHBOR_DISTANCE = 2.0


@dataclass
class PredictionResult:
    label: Optional[str]
    confidence: float  # 0..1, vote fraction among nearest neighbors
    needs_confirmation: bool
    reason: str  # "ok" | "low_confidence" | "no_match" | "not_trained"


@dataclass
class SignKNNClassifier:
    """Wraps sklearn's KNeighborsClassifier with enrollment bookkeeping.

    Examples are kept as raw (vector, label) pairs so the model can always
    be rebuilt from scratch (sklearn's KNN has no meaningful partial_fit).
    """

    k_neighbors: int = 3
    confidence_threshold: float = 0.66
    _vectors: list[np.ndarray] = field(default_factory=list)
    _labels: list[str] = field(default_factory=list)
    _model: Optional[KNeighborsClassifier] = field(default=None, repr=False)

    def add_example(self, vector: np.ndarray, label: str) -> None:
        self._vectors.append(np.asarray(vector, dtype=np.float64))
        self._labels.append(label)
        self._model = None  # stale; rebuild lazily on next predict/fit

    def label_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for label in self._labels:
            counts[label] = counts.get(label, 0) + 1
        return counts

    def is_trainable(self) -> bool:
        counts = self.label_counts()
        trainable_labels = [l for l, c in counts.items() if c >= MIN_EXAMPLES_PER_LABEL]
        return len(trainable_labels) >= 2

    def fit(self) -> None:
        if not self.is_trainable():
            self._model = None
            return
        counts = self.label_counts()
        keep_labels = {l for l, c in counts.items() if c >= MIN_EXAMPLES_PER_LABEL}
        X, y = [], []
        for vec, label in zip(self._vectors, self._labels):
            if label in keep_labels:
                X.append(vec)
                y.append(label)
        k = min(self.k_neighbors, len(X))
        model = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
        model.fit(np.array(X), np.array(y))
        self._model = model

    def predict(self, vector: np.ndarray) -> PredictionResult:
        if self._model is None:
            self.fit()
        if self._model is None:
            return PredictionResult(None, 0.0, True, "not_trained")

        vector = np.asarray(vector, dtype=np.float64).reshape(1, -1)
        distances, indices = self._model.kneighbors(vector)
        nearest_distance = float(distances[0][0])
        if nearest_distance > MAX_NEIGHBOR_DISTANCE:
            return PredictionResult(None, 0.0, True, "no_match")

        proba = self._model.predict_proba(vector)[0]
        classes = self._model.classes_
        best_idx = int(np.argmax(proba))
        label = str(classes[best_idx])
        confidence = float(proba[best_idx])

        if confidence < self.confidence_threshold:
            return PredictionResult(label, confidence, True, "low_confidence")
        return PredictionResult(label, confidence, False, "ok")

    def to_state(self) -> dict:
        return {
            "k_neighbors": self.k_neighbors,
            "confidence_threshold": self.confidence_threshold,
            "vectors": [v.tolist() for v in self._vectors],
            "labels": self._labels,
        }

    @classmethod
    def from_state(cls, state: dict) -> "SignKNNClassifier":
        clf = cls(
            k_neighbors=state.get("k_neighbors", 3),
            confidence_threshold=state.get("confidence_threshold", 0.66),
        )
        clf._vectors = [np.array(v, dtype=np.float64) for v in state.get("vectors", [])]
        clf._labels = list(state.get("labels", []))
        return clf
