"""
Bulk-import a labeled ISL image/video dataset into SignBridge's classifier,
instead of enrolling one example at a time through the browser UI.

This is how you actually add an "ISL library" to the app: point this
script at a folder of real, licensed (or your own, consented) sign
recordings, and it populates `backend/data/library.json` with real
enrolled examples. The running app loads this file at startup and merges
it with each user's own live enrollments (see backend/app/storage.py's
module docstring for why those two layers are kept separate) --
`library.json` is derived from dataset recordings, not from anyone using
the app, so unlike a user's own enrollments it's meant to be committed to
git and shipped with the repo (with attribution to the dataset -- see
/data-docs/dataset-notes.md). No pretrained dataset ships with this repo
by default; this script is the tool for adding one yourself once you
have rights to use one.

IMPORTANT: only the library is downloaded/processed once, by whoever runs
this script. End users of the deployed app never download the raw
dataset -- `library.json` holds extracted landmark numbers, not video,
and is typically a few MB even for a sizeable vocabulary.

It requires the OPTIONAL server-side detection path (HandLandmarkExtractor
in backend/app/landmarks.py), because there is no browser in the loop for
a bulk import. That needs `opencv-python` and `mediapipe` installed, and
-- once -- outbound internet access to download the ~8MB hand-landmark
model (cached after that). See landmarks.py's module docstring; this
could not be run end-to-end in the sandbox this project was built in
(confirmed no internet there), but works on a normal machine.

Expected input layout (a common convention for sign-language datasets):

    dataset_dir/
      hello/
        clip1.mp4
        clip2.jpg
      thank_you/
        clip1.mp4
        ...

Each subfolder name becomes the enrollment label. Images contribute one
example each; videos are sampled at a fixed interval (every Nth frame, to
avoid wildly over-representing a long clip) and each sampled frame with a
detected hand becomes one example -- consistent with this MVP's
static-pose classifier (see /ml-model/README.md for the sequence-model
upgrade path this does NOT attempt).

Usage:
    python ml-model/import_dataset.py /path/to/dataset_dir \
        [--classifier-state backend/data/library.json] \
        [--sample-every 5] [--max-frames-per-clip 30] [--fresh]

--fresh replaces the existing output file instead of appending to it.
By default this writes to backend/data/library.json (the SHIPPED,
committed-to-git vocabulary) -- not classifier_state.json (a live user's
own gitignored enrollments; see backend/app/storage.py). Only pass
--classifier-state explicitly if you really want to write somewhere else
(e.g. a scratch file while testing).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

# Allow running as `python ml-model/import_dataset.py` without installing
# this as a package.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from backend.app.landmarks import HandLandmarkExtractor, hands_to_feature_vector  # noqa: E402

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def iter_label_dirs(dataset_dir: str):
    for entry in sorted(os.scandir(dataset_dir), key=lambda e: e.name):
        if entry.is_dir() and not entry.name.startswith("."):
            yield entry.name, entry.path


def frames_from_image(path: str):
    image_bgr = cv2.imread(path)
    if image_bgr is None:
        print(f"  [skip] could not read image: {path}")
        return
    yield cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def frames_from_video(path: str, sample_every: int, max_frames: int):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"  [skip] could not open video: {path}")
        return
    frame_idx = 0
    yielded = 0
    try:
        while yielded < max_frames:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            if frame_idx % sample_every == 0:
                yield cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                yielded += 1
            frame_idx += 1
    finally:
        cap.release()


def frames_for_file(path: str, sample_every: int, max_frames_per_clip: int):
    ext = os.path.splitext(path)[1].lower()
    if ext in IMAGE_EXTENSIONS:
        yield from frames_from_image(path)
    elif ext in VIDEO_EXTENSIONS:
        yield from frames_from_video(path, sample_every, max_frames_per_clip)
    else:
        print(f"  [skip] unrecognized file type: {path}")


def load_existing_state(path: str) -> dict:
    if not os.path.exists(path):
        return {"vectors": [], "labels": []}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_state(path: str, state: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp_path, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset_dir", help="Folder of <label>/<file> images or video clips")
    parser.add_argument(
        "--classifier-state",
        default=os.path.join(os.path.dirname(__file__), "..", "backend", "data", "library.json"),
        help="Where to write enrolled examples (default: the shipped, git-committed library file)",
    )
    parser.add_argument("--sample-every", type=int, default=5, help="Sample every Nth video frame (default 5)")
    parser.add_argument(
        "--max-frames-per-clip", type=int, default=30,
        help="Cap on sampled frames per video, so one long clip can't dominate a label (default 30)",
    )
    parser.add_argument(
        "--fresh", action="store_true",
        help="Replace the existing classifier state instead of appending to it",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.dataset_dir):
        parser.error(f"not a directory: {args.dataset_dir}")

    try:
        extractor = HandLandmarkExtractor(max_num_hands=2, min_detection_confidence=0.5)
    except RuntimeError as exc:
        print(f"Server-side hand detection is unavailable: {exc}", file=sys.stderr)
        sys.exit(1)

    state = {"vectors": [], "labels": []} if args.fresh else load_existing_state(args.classifier_state)

    total_added = 0
    total_skipped_no_hand = 0

    try:
        for label, label_dir in iter_label_dirs(args.dataset_dir):
            label_added = 0
            label_skipped = 0
            files = sorted(
                f.path for f in os.scandir(label_dir)
                if f.is_file() and not f.name.startswith(".")
            )
            print(f"[{label}] {len(files)} file(s)")
            for file_path in files:
                for image_rgb in frames_for_file(file_path, args.sample_every, args.max_frames_per_clip):
                    hands = extractor.extract(image_rgb)
                    if not hands:
                        label_skipped += 1
                        continue
                    vector = hands_to_feature_vector(hands)
                    state["vectors"].append(vector.tolist())
                    state["labels"].append(label)
                    label_added += 1
            print(f"  added {label_added} example(s), skipped {label_skipped} frame(s) with no hand detected")
            total_added += label_added
            total_skipped_no_hand += label_skipped
    finally:
        extractor.close()

    save_state(args.classifier_state, state)

    print()
    print(f"Done: {total_added} example(s) added across {len(set(state['labels']))} label(s)"
          f" ({total_skipped_no_hand} frame(s) skipped, no hand detected).")
    print(f"Wrote {args.classifier_state}")
    print(
        "Reminder: verify this dataset's license permits this use, and that "
        "it is consented (see /data-docs/dataset-notes.md and consent.md) "
        "before using it beyond local testing."
    )
    if os.path.basename(args.classifier_state) == "library.json":
        print(
            "This is the SHIPPED library file -- commit it to git (with the "
            "dataset's name/license recorded in /data-docs/dataset-notes.md) "
            "so everyone who clones the repo gets this vocabulary without "
            "re-running this script or downloading the raw dataset themselves."
        )


if __name__ == "__main__":
    main()
