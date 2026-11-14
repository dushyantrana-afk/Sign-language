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
| On-device/offline inference | ⚠️ partial — hand detection runs in-browser (on-device); classification is a local backend call, not cloud |
| Sign-language avatar output | ❌ listed in the deck as a future extension, not MVP scope |
| Pretrained ISL dataset / full ISL vocabulary | ❌ — see [Why no pretrained dataset](#why-no-pretrained-dataset) |

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

This differs from the deck's exact architecture diagram in one place,
deliberately: the deck's "AI MODEL: hand landmarks" step runs in the
browser (MediaPipe Hands JS) rather than as a server-side OpenCV/MediaPipe
call. See the top of `backend/app/landmarks.py` for the full reasoning —
short version: the environment this was built in has no network access to
fetch model weights and an incomplete MediaPipe Python install, and
keeping raw video on-device is a genuine privacy win anyway (see
`data-docs/consent.md`).

## How recognition works

There is **no pretrained sign classifier**. Instead:

1. **Enroll**: on the "Enroll a sign" tab, type a label (e.g. `hello`),
   show the sign to the camera, and click "Capture example" 3+ times.
2. Once at least **2 labels have 3+ examples each**, live recognition
   turns on.
3. **Recognize**: the "Live recognition" tab sends the current hand
   landmarks to the backend roughly once a second; the backend finds the
   nearest enrolled example (K-nearest-neighbors over normalized landmark
   vectors) and returns a label + confidence.
4. **Confirm**: if confidence is low, or the gesture doesn't closely match
   anything enrolled, the UI asks you to confirm or correct it instead of
   stating the guess as fact — this mirrors the deck's "never silently
   convert uncertain predictions into definitive statements" rule
   (slide 9) and is enforced in `classifier.py`'s `needs_confirmation`
   logic, not just in the UI copy.

### Why no pretrained dataset

The deck requires using "licensed/public datasets only where the license
permits," with source and consent documented (slide 13). No such
ISL-specific dataset has been identified and licensed yet. Rather than
either blocking on that or quietly using something unlicensed, this MVP
recognizes whatever small vocabulary a user enrolls live — see
`data-docs/dataset-notes.md` for the full reasoning and the upgrade path
once real licensed/consented data exists (`ml-model/README.md`).

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
