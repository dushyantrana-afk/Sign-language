"""
Offline training skeleton for the future, dataset-backed sign classifier.

This is intentionally NOT imported by the live backend (backend/app/main.py
uses the live-enrollment SignKNNClassifier instead — see /ml-model/README.md
for why). Run this manually once a larger, consented dataset of exported
landmark vectors exists, to train and evaluate a baseline that could later
replace the live-enrollment model.

Expected input format: a JSON file shaped like:

    [
      {"label": "hello", "signer_id": "s01", "vector": [0.1, 0.2, ...]},
      {"label": "hello", "signer_id": "s02", "vector": [0.1, 0.3, ...]},
      ...
    ]

where each "vector" is a 127-dim feature vector produced by
`backend.app.landmarks.hands_to_feature_vector`. `signer_id` is optional;
when present, evaluation splits by signer (per the deck's "split by signer
to reduce data leakage" guidance) instead of a plain random split.

Usage:
    python ml-model/train_classifier.py path/to/exported_examples.json
"""
from __future__ import annotations

import json
import sys

import numpy as np
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report


def load_examples(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    with open(path, "r", encoding="utf-8") as f:
        records = json.load(f)
    if not records:
        raise ValueError("no examples found in " + path)

    X = np.array([r["vector"] for r in records], dtype=np.float64)
    y = np.array([r["label"] for r in records])
    signer_ids = None
    if all("signer_id" in r for r in records):
        signer_ids = np.array([r["signer_id"] for r in records])
    return X, y, signer_ids


def split(X, y, signer_ids):
    if signer_ids is not None and len(set(signer_ids)) > 1:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=0)
        train_idx, test_idx = next(splitter.split(X, y, groups=signer_ids))
        print("Using signer-grouped split (test signers are unseen in training).")
        return X[train_idx], X[test_idx], y[train_idx], y[test_idx]
    print(
        "WARNING: no usable signer_id column found — falling back to a plain "
        "random split. Reported accuracy will likely overstate real-world "
        "generalization to new signers. See /ml-model/README.md."
    )
    return train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)


def main(path: str) -> None:
    X, y, signer_ids = load_examples(path)
    X_train, X_test, y_train, y_test = split(X, y, signer_ids)

    k = min(5, len(X_train))
    model = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, zero_division=0))

    print(
        "This is a baseline, not a final model. See /ml-model/README.md step 4 "
        "for how to extend this to sequence/motion data instead of single "
        "static-frame vectors."
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])
