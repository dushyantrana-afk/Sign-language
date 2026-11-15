# SignBridge AI — hackathon MVP

A two-way communication aid for deaf and hard-of-hearing users, built from
the `SignBridge_AI_Hackathon_Pitch` deck's scope: camera-based hand-sign
recognition for a **small, live-enrolled vocabulary**, plus speech↔text
support. This is a working prototype matching the deck's own description —
"a focused, testable prototype, not a claim to translate every sign
language or every conversation."

## What's actually implemented

| Deck feature | Status |
|---|---|
| Live sign recognition (small vocabulary) | ✅ working — see [How recognition works](#how-recognition-works) |
| Text/speech output for a prediction | ✅ browser `speechSynthesis` |
| Speech → captions | ✅ browser `SpeechRecognition` |
| Text-to-speech for typed messages | ✅ |
| Phrase shortcuts | ✅ |
| Correction & feedback, stored only with consent | ✅ |
| User confirmation before trusting a low-confidence prediction | ✅ — see `needs_confirmation` |
| On-device/offline inference | ⚠️ partial — hand detection runs in-browser by default (on-device); classification is a local backend call, not cloud. An optional server-side detection mode also exists — see below |
| Sign-language avatar output | ❌ listed in the deck as a future extension, not MVP scope |
| Pretrained ISL dataset / full ISL vocabulary | ❌ bundled, but ✅ importable — see [Adding a real ISL vocabulary](#adding-a-real-isl-vocabulary) |

## Architecture

```
Browser (frontend/)                          Backend (backend/app/)
┌─────────────────────────────┐              ┌───────────────────────────┐
│ Camera → MediaPipe Hands JS  │  landmarks   │ landmarks.py              │
│ (hand landmark detection,    │  (JSON, not  │  normalize + build a      │
│  runs on-device in-browser)  │ ──image──>   │  fixed-length feature     │
│                               │              │  vector                   │
│ SpeechSynthesis (TTS)        │              │                           │
│ SpeechRecognition (STT)      │              │ classifier.py             │
│   -- both local to browser,  │              │  KNN over enrolled        │
│      nothing sent to server  │              │  examples, with a         │
│                               │  prediction  │  confidence threshold and │
│ UI: enroll / recognize /     │ <── JSON ──  │  "no match" rejection     │
│     speech tabs, confirm     │              │                           │
│     step, phrase shortcuts   │              │ storage.py                │
└─────────────────────────────┘              │  JSON-file persistence    │
                                              │                           │
                                              │ main.py (Flask)           │
                                              │  /api/enroll /predict     │
                                              │  /api/vocabulary          │
                                              │  /api/feedback /reset     │
                                              └───────────────────────────┘
```

This differs from the deck's exact architecture diagram in one place: the
deck's "AI MODEL: hand landmarks" step defaults to running in the browser
(MediaPipe Hands JS) rather than as a server-side OpenCV/MediaPipe call.
Keeping raw video on-device by default is a genuine privacy win (see
`data-docs/consent.md`). **Both `/api/enroll` and `/api/predict` also
accept a server-side mode** — send `{"image": "data:..."}` instead of
`{"hands": [...]}` — which runs real detection on the backend via
MediaPipe's Tasks API (`HandLandmarkExtractor` in `backend/app/landmarks.py`).
The frontend's "Detect hands on the server" checkbox switches to this
mode, and falls back to it automatically if the browser can't load the
MediaPipe Hands JS library. Server-side detection needs the backend
machine to have normal internet access once, to download and cache a
small model file — the sandbox this project was originally developed in
specifically couldn't (confirmed via a direct network test), so that path
is implemented and unit-tested for its error handling but not exercised
end-to-end there. On a normal machine it just works; see
`/api/capabilities` and the error message you'd get otherwise.

### Adding a real ISL vocabulary

No pretrained ISL vocabulary ships in this repo *by default* (see
[Why no pretrained dataset yet](#why-no-pretrained-dataset-yet)), but you
don't have to enroll signs one at a time through the browser either, and
end users never download a raw dataset. Once you have a licensed or
self-recorded dataset of labeled sign clips/images, run:

```bash
python ml-model/import_dataset.py /path/to/dataset_dir
```

This bulk-populates `backend/data/library.json` using the server-side
detection path above. **This file holds extracted landmark numbers, not
video** — typically a few MB even for a sizeable vocabulary, since the
raw footage is discarded right after detection. It's a *separate* file
from a user's own live enrollments (`classifier_state.json`): the app
merges both at startup, but only `library.json` is meant to be committed
to git (see `backend/app/storage.py`'s module docstring for why) and
`/api/reset` never touches it. See `ml-model/README.md` and
`data-docs/dataset-notes.md` (which lists candidate public ISL datasets
— INCLUDE and ISL-CSLTR are both confirmed CC BY 4.0, no access request
needed, as of November 2026) for details.

## How recognition works

1. **Library (optional)**: if `backend/data/library.json` exists (built
   via the importer above), its signs are available the moment the app
   starts — no enrollment needed for those.
2. **Enroll (always available)**: on the "Enroll a sign" tab, type a
   label (e.g. `hello`), show the sign to the camera, and click "Capture
   example" 3+ times, to add your own signs on top of (or instead of) the
   library.
3. Once at least **2 labels (library + your own, combined) have 3+
   examples each**, live recognition turns on.
4. **Recognize**: the "Live recognition" tab sends the current hand
   landmarks to the backend roughly once a second; the backend finds the
   nearest enrolled example (K-nearest-neighbors over normalized landmark
   vectors, across both the library and your own enrollments) and returns
   a label + confidence.
5. **Confirm**: if confidence is low, or the gesture doesn't closely match
   anything enrolled, the UI asks you to confirm or correct it instead of
   stating the guess as fact — this mirrors the deck's "never silently
   convert uncertain predictions into definitive statements" rule
   (slide 9) and is enforced in `classifier.py`'s `needs_confirmation`
   logic, not just in the UI copy.

### Why no pretrained dataset yet

The deck requires using "licensed/public datasets only where the license
permits," with source and consent documented (slide 13), and recording
that is a deliberate step (see `data-docs/dataset-notes.md`) — so this
repo doesn't ship `library.json` until the team actually runs the
importer against a chosen dataset and commits the result. Until then,
this MVP recognizes whatever small vocabulary a user enrolls live.

## Running it

Requires Python 3.9+ with the packages in `requirements.txt` (Flask,
numpy, scikit-learn). A normal internet connection is needed in the
**browser** (not the backend) to load the MediaPipe Hands JS library from
its CDN and the speech APIs are built into the browser.

```bash
pip install -r requirements.txt
python -m backend.app.main
# then open http://localhost:5000 in Chrome (speech APIs need a Chromium-based browser)
```

Click "Start camera" (camera only turns on when you choose to), go to
**Enroll a sign**, enroll 2+ signs with 3+ examples each, then switch to
**Live recognition** and click "Start recognizing."

To wipe all enrolled data: the "Delete all enrolled data" button on the
Enroll tab, or `POST /api/reset`.

## Running the tests

```bash
pip install -r requirements.txt
python -m pytest tests/ -v
```

31 tests cover landmark normalization (translation/scale invariance),
the classifier (confident match / low-confidence / no-match / state
persistence), and the API endpoints end-to-end with synthetic landmark
payloads (no real camera or browser needed to run these).

## Repository layout

```
backend/app/
  landmarks.py      # normalize MediaPipe landmarks -> fixed feature vector
  classifier.py      # KNN classifier with confidence/confirmation logic
  storage.py          # JSON-file persistence (enrolled signs, feedback)
  main.py              # Flask API + serves frontend/
frontend/
  index.html, style.css, app.js   # webcam UI, MediaPipe Hands JS, speech APIs
data-docs/
  vocabulary.md      # starter sign list (draft, needs ISL-signer review)
  consent.md          # what's captured, when, and what isn't stored
  dataset-notes.md     # why there's no bundled dataset + licensing notes
ml-model/
  README.md           # upgrade path: live-enrollment KNN -> trained sequence model
  train_classifier.py # offline training skeleton for once real data exists
tests/                 # pytest suite (31 tests, see above)
```

## Honest limitations (don't oversell this in a demo)

- Recognizes **static hand shapes only** — not motion, not facial
  expression/non-manual markers that many real signs depend on. See
  `ml-model/README.md` for the path to a sequence model.
- The KNN model does **not generalize across signers** — it is trained on
  whatever was enrolled in the current session, so it will recognize the
  enroller's own hand far better than a stranger's. This is expected and
  acceptable for a hackathon demo; it is not acceptable to claim as
  "working for everyone" (see slide 13 of the deck, and `dataset-notes.md`).
- No per-user/session scoping yet — everyone using one running instance
  shares the same enrolled vocabulary.
- No database (just JSON files) and no encryption at rest — fine for a
  local demo, not for a real pilot; see `data-docs/consent.md`.
- The confidence/distance thresholds in `classifier.py` are starting
  heuristics, not calibrated against real usage data.
