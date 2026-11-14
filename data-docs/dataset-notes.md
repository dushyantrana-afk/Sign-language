# Dataset notes

## This MVP ships with no dataset, on purpose

The pitch deck is explicit that any dataset used must be "licensed/public...
where the license permits your intended use," with source and consent
documented. Rather than bundle a scraped or unlicensed ISL dataset to make
a demo look more impressive, this build uses **live, few-shot enrollment**:
a user records 3+ examples of their own hand shape for each sign, and the
classifier (`backend/app/classifier.py`) is a K-nearest-neighbors model
over those enrolled examples — no external training data required.

This is an honest trade-off, not a shortcut:

- **Pro:** zero licensing risk, zero unconsented biometric data, works
  immediately for a demo, and matches the deck's "small verified
  vocabulary" framing exactly.
- **Con:** it does not generalize across signers. Signs enrolled by one
  person's hand will recognize that person much better than a stranger's.
  This is *expected* for the hackathon MVP stage and should be stated
  plainly in any demo, not glossed over.

## Adding an ISL "library" (a real starter vocabulary)

Two separate upgrades, and it's worth not conflating them:

1. **Bulk-populate the existing per-session KNN model** with a real
   vocabulary, so the app ships with signs already enrolled instead of
   needing live enrollment every run. Use `/ml-model/import_dataset.py`:
   point it at a folder of `<label>/<clip-or-image>` files (from a
   licensed dataset you've downloaded, or your own consented recordings)
   and it runs server-side hand-landmark detection on each file and
   writes the resulting examples straight into
   `backend/data/classifier_state.json` — the exact file the live app
   reads. This needs the optional server-side detection path (see
   [Landmark detection: client vs. server](#landmark-detection-client-vs-server)
   below) and, once, outbound internet to fetch the detection model.
2. **Train a signer-generalizing sequence model** instead of KNN, once
   there's enough data — that's `/ml-model/train_classifier.py` /
   `/ml-model/README.md`, a bigger undertaking than (1).

Start with (1); it's the realistic "add an ISL library" step for now.

### Candidate ISL datasets (not bundled here — check license before use)

No ISL dataset ships with this repo. If you want to populate the app with
real ISL vocabulary via `import_dataset.py`, these are publicly referenced
ISL datasets to look into — **check each one's actual license and terms
yourself before use; this list is a starting point for research, not a
license clearance:**

- **INCLUDE** (AI4Bharat / IIT Madras, with IIIT-B) — isolated-sign ISL
  video dataset, ~4,287 videos across 263 signs from multiple signers.
  Look up current access terms from AI4Bharat before use.
- **ISL-CSLTR** — an Indian Sign Language continuous/isolated dataset
  referenced in ISL research literature.
- **INCLUDE-50** — a smaller subset of INCLUDE, useful for a first
  small-vocabulary pass matching this MVP's "10-20 signs" scope (deck
  slide 14).

For each: confirm the license permits this project's intended use
(a hackathon demo vs. anything resembling a product), record the dataset
name, version/date, and license terms **here** once one is actually
selected, and keep that record up to date — per `/data-docs/vocabulary.md`
and the deck's own "document dataset source and consent" requirement.
**Do not claim an ISL dataset has been secured until one is**, with its
license checked and recorded in this file.

## Landmark detection: client vs. server

`backend/app/landmarks.py` supports two ways to get from a camera frame to
hand landmarks:

- **Browser-side (the live app's default)** — MediaPipe Hands JS runs in
  the user's browser; only landmark coordinates are sent to the backend.
  No dataset tool needed; this is what `frontend/app.js` uses day-to-day.
- **Server-side (optional, used by `import_dataset.py` and available as a
  toggle in the live app)** — `HandLandmarkExtractor` runs MediaPipe's
  Tasks API on a frame/image file directly. This is **required** for bulk
  dataset import, since there's no browser in that loop.

Server-side detection downloads a small (~8MB) model file from Google's
MediaPipe model hosting the first time it's used, caching it under
`backend/models/`. That download needs a normal internet connection on
whichever machine runs the backend or the import script — confirmed
unavailable in the sandbox this project was originally built in (that
network blocks `storage.googleapis.com`), but this is a sandbox-specific
limitation, not a design flaw: a normal laptop or server will not have the
same restriction. If you hit a `server_side_detection_unavailable` error,
it means that *specific machine* can't reach the model host right now —
see the error message for the exact URL and directory involved.

## Data this app itself generates

- `backend/data/classifier_state.json` — enrolled (landmark-vector, label)
  pairs. This *is* personal/biometric-adjacent data for whoever enrolled
  it. Delete via the app's "Delete all enrolled data" button before
  sharing a machine or repo state with anyone else.
- `backend/data/feedback_log.json` — consented correction entries only
  (see `consent.md`).

Both files are gitignored; they should never be committed.
