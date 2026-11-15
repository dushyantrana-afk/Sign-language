"""
Minimal JSON-file persistence for the MVP.

The pitch deck lists SQLite/PostgreSQL for "minimal settings / consented
feedback" storage. A single JSON file is the honest MVP substitute: it is
easy to inspect, easy to delete entirely (data minimization), and avoids
pulling in a database for a hackathon prototype. Swapping this module for
a real SQLite-backed store later does not require touching the API layer,
since everything goes through the functions below.

Two separate enrollment-data files, with different git/privacy treatment:

- `library.json` -- a shipped, pretrained vocabulary built OFFLINE (by a
  maintainer, via `ml-model/import_dataset.py`) from a licensed/public
  dataset. Not personal data -- it's derived from dataset recordings, not
  from anyone using this app -- so it's meant to be committed to git and
  distributed with the repo. The running app never writes to this file;
  only the import script does.
- `classifier_state.json` -- live examples a *user of this running
  instance* enrolls through the browser. This IS personal/biometric-
  adjacent data (see /data-docs/consent.md), so it's gitignored, and
  `/api/reset` deletes it without touching the shipped library.

main.py's classifier is built from BOTH at startup (library + user,
merged), but only ever persists the user layer back to disk.
"""
from __future__ import annotations

import json
import os
import threading
import time
from typing import Any

_LOCK = threading.Lock()

DEFAULT_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
LIBRARY_STATE_FILE = "library.json"
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


def load_library_state() -> dict:
    """The shipped, pretrained vocabulary (see module docstring). Read-only
    from the running app's point of view -- only import_dataset.py writes
    this file."""
    return load_json(LIBRARY_STATE_FILE, {"vectors": [], "labels": []})


def load_classifier_state() -> dict:
    """A user's own live enrollments on top of the shipped library."""
    return load_json(CLASSIFIER_STATE_FILE, {"vectors": [], "labels": []})


def save_classifier_state(state: dict) -> None:
    save_json(CLASSIFIER_STATE_FILE, state)


def reset_library() -> None:
    """Delete the shipped library too. NOT wired to any API route by
    default -- /api/reset deliberately only clears the user layer (see
    reset_all()). This is here for a maintainer rebuilding the library
    from scratch locally, not for the running app to call."""
    path = _path(LIBRARY_STATE_FILE)
    if os.path.exists(path):
        os.remove(path)


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
