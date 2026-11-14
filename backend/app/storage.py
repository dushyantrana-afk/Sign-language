"""
Minimal JSON-file persistence for the MVP.

The pitch deck lists SQLite/PostgreSQL for "minimal settings / consented
feedback" storage. A single JSON file is the honest MVP substitute: it is
easy to inspect, easy to delete entirely (data minimization), and avoids
pulling in a database for a hackathon prototype. Swapping this module for
a real SQLite-backed store later does not require touching the API layer,
since everything goes through the functions below.
"""
from __future__ import annotations

import json
import os
import threading
import time
from typing import Any

_LOCK = threading.Lock()

DEFAULT_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
CLASSIFIER_STATE_FILE = "classifier_state.json"
FEEDBACK_FILE = "feedback_log.json"


def _data_dir() -> str:
    data_dir = os.environ.get("SIGNBRIDGE_DATA_DIR", DEFAULT_DATA_DIR)
    os.makedirs(data_dir, exist_ok=True)
    return data_dir


def _path(filename: str) -> str:
    return os.path.join(_data_dir(), filename)


def load_json(filename: str, default: Any) -> Any:
    path = _path(filename)
    if not os.path.exists(path):
        return default
    with _LOCK, open(path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return default


def save_json(filename: str, data: Any) -> None:
    path = _path(filename)
    tmp_path = path + ".tmp"
    with _LOCK:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, path)


def load_classifier_state() -> dict:
    return load_json(CLASSIFIER_STATE_FILE, {"vectors": [], "labels": []})


def save_classifier_state(state: dict) -> None:
    save_json(CLASSIFIER_STATE_FILE, state)


def append_feedback(entry: dict) -> None:
    """Append one consented feedback/correction entry. Best-effort, append-only."""
    entries = load_json(FEEDBACK_FILE, [])
    entry = dict(entry)
    entry["timestamp"] = time.time()
    entries.append(entry)
    save_json(FEEDBACK_FILE, entries)


def load_feedback() -> list[dict]:
    return load_json(FEEDBACK_FILE, [])


def reset_all() -> None:
    """Delete all enrolled signs and feedback. Used by the demo 'reset' control
    and by data-minimization requests ("delete my data")."""
    for filename in (CLASSIFIER_STATE_FILE, FEEDBACK_FILE):
        path = _path(filename)
        if os.path.exists(path):
            os.remove(path)
