# ML model — future extension path

## What exists today

`backend/app/classifier.py` is a K-nearest-neighbors model over hand-landmark
feature vectors produced by `backend/app/landmarks.py`. It is trained from
whatever a user enrolls live in the app (see `/data-docs/dataset-notes.md`
for why). It handles **static hand shapes** well. It does **not** model:

- motion/trajectory (many signs are defined by movement, not just a
  final hand shape)
- two-hand *relative* timing (which hand moves first/last)
- non-manual markers (facial expression, head tilt) that change meaning
  in real sign languages

This matches the deck's own caution (slide 13): "a hand-shape-only demo is
a limited gesture recognizer — not full sign-language translation."

## Upgrade path: a trained sequence model

Once a consented, signer-diverse dataset exists (see dataset-notes.md),
replace the per-session KNN with a model trained offline on landmark
*sequences* (a short clip's worth of frames, not one still frame):

1. **Collect sequences, not single frames.** Extend the enrollment capture
   in `frontend/app.js` to record N consecutive frames (e.g. 1 second at
   10fps) per example, and extend `/api/enroll` to accept a list of images
   or stream frames over a short window.
2. **Feature extraction stays the same per-frame** — reuse
   `hands_to_feature_vector()` from `backend/app/landmarks.py` for every
   frame in the sequence, giving a `(T, 127)` array per example.
3. **Split by signer, not just randomly**, for train/validation/test, so
   the reported accuracy reflects generalization to a new person rather
   than memorizing one signer's examples (the deck calls this out
   explicitly in slide 13 — "split train, validation and test sets by
   signer to reduce data leakage").
4. **Train a sequence model** (e.g. an LSTM/GRU or a small Transformer in
   TensorFlow/Keras or PyTorch, per the deck's tech stack) on the `(T, 127)`
   sequences. For a first iteration, even a simple approach — averaging or
   max-pooling each sequence down to one vector and reusing
   `SignKNNClassifier`/`KNeighborsClassifier` — is a reasonable baseline
   before investing in a recurrent model.
5. **Export** the trained model's weights (or, for the sklearn baseline,
   a pickled/joblib model) and swap it in behind the same
   `SignKNNClassifier.predict()` interface so `backend/app/main.py` does
   not need to change — only `get_classifier()` would load the trained
   model instead of the live-enrolled one.
6. **Re-run the confidence/confirmation logic as-is.** The
   "never silently convert uncertain predictions" rule
   (`needs_confirmation`, `MAX_NEIGHBOR_DISTANCE` / probability threshold
   in `classifier.py`) should carry over to whatever model replaces KNN —
   it is a product requirement, not an artifact of this specific algorithm.

## `train_classifier.py`

A runnable skeleton for step 4's simplest baseline: train an
`sklearn.neighbors.KNeighborsClassifier` (swap in any sklearn classifier)
on landmark vectors exported from `backend/data/classifier_state.json`,
evaluate with a signer-aware split if a `signer_id` column is present, and
save the trained model as a baseline to compare future sequence models
against. It is deliberately not wired into the live backend — it is a
training/evaluation script for when there is enough real data to make
training meaningful, not a replacement for the live-enrollment MVP.
