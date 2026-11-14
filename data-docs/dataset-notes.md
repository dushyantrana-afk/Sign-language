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

## Upgrade path, if/when a real dataset is available

If the team gets access to a properly licensed or newly collected
(consented, signer-diverse) ISL dataset, `/ml-model/train_classifier.py`
shows how to train a signer-generalizing classifier offline from exported
landmark vectors, replacing the per-session KNN model. See
`/ml-model/README.md`.

Known public sign-language datasets exist (e.g. for ASL and a few other
sign languages); as of this writing the team has not identified and
licensed an ISL-specific dataset suitable for this project. **Do not claim
an ISL dataset has been secured until one is, with its license checked and
recorded here.**

## Data this app itself generates

- `backend/data/classifier_state.json` — enrolled (landmark-vector, label)
  pairs. This *is* personal/biometric-adjacent data for whoever enrolled
  it. Delete via the app's "Delete all enrolled data" button before
  sharing a machine or repo state with anyone else.
- `backend/data/feedback_log.json` — consented correction entries only
  (see `consent.md`).

Both files are gitignored; they should never be committed.
