# Consent and data handling

This MVP follows the deck's "Trust by Design" commitments. Concretely, in
this codebase:

## What is captured, and when

- The camera/microphone are **never** started automatically. `frontend/app.js`
  only requests camera access after the user clicks **"Start camera"**
  (`getUserMedia` is called from that click handler, nowhere else).
- Speech-to-text and text-to-speech run **entirely in the browser** via the
  Web Speech API (`webkitSpeechRecognition` / `speechSynthesis`). No audio
  is ever sent to the backend.
- The only images sent to the backend are single still frames, captured on
  demand:
  - one per **enrollment** click (`/api/enroll`)
  - one per **recognition tick** while live recognition is explicitly
    turned on (`/api/predict`), roughly once per second
- **No video is stored by default.** The backend (`backend/app/main.py`)
  decodes each frame in memory, extracts a hand-landmark feature vector,
  and discards the image. Only the numeric landmark vector (not the
  picture) is written to disk, and only for enrollment examples.

## Consent for feedback/corrections

- `/api/feedback` stores a correction (predicted label vs. the label the
  user says is actually correct) **only when the request includes
  `"consent": true`.** The frontend ties this to an explicit, unchecked-by-
  default checkbox ("I consent to this correction being stored...") — see
  `frontend/index.html` / `app.js`. Declining consent still lets the user
  correct the on-screen label; it just isn't written to `feedback_log.json`.

## Deletion

- The "Delete all enrolled data" button (`/api/reset`) deletes *this
  user's own* enrollment store (`backend/data/classifier_state.json`) and
  the feedback log from disk.
- It deliberately does **not** delete `backend/data/library.json` — the
  shipped, pretrained vocabulary a maintainer built offline from a
  licensed dataset via `ml-model/import_dataset.py` (see
  `/data-docs/dataset-notes.md`). That file isn't personal data collected
  from this app's users, so clearing it out from under everyone using a
  running instance would be surprising, not protective. A maintainer who
  genuinely wants to rebuild or clear it can call
  `storage.reset_library()` directly; it isn't exposed as an API route.

## What this MVP does **not** yet implement (be upfront about this with users)

- No login/account system — enrolled signs are shared across anyone using
  the same running instance. A real pilot needs per-user or per-session
  enrollment scoping before this leaves "demo" status.
- No on-device/offline inference yet — frames are sent to the local backend
  process. The deck's "process on-device where possible" goal is not met
  by this MVP; it is scoped to a backend on the same machine/network as the
  demo, not a cloud service receiving footage from third parties.
- No encryption at rest for `backend/data/*.json` — fine for a local
  hackathon demo, not fine for storing anyone's real biometric-adjacent
  data in a pilot. Treat anything beyond a live demo as needing a proper
  data-protection review first.
