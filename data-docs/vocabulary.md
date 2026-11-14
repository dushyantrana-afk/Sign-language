# Starter vocabulary

This MVP ships with **no pretrained signs**. There is no bundled dataset —
every sign a user sees recognized is one they (or a demo operator) enrolled
live through the app, by recording a few camera examples of their own hand
shape for that label (see [dataset-notes.md](dataset-notes.md) for why).

The list below is a *suggested* starting vocabulary for a demo — common,
high-value, low-ambiguity signs for first contact between a sign-language
user and a non-signer. **It is a draft, not a verified ISL reference.**
Per the pitch deck's own caution, it must be reviewed with fluent ISL
signers and accessibility partners before being treated as correct or
used in any real pilot.

| Label (enrollment key) | Spoken/written meaning | Why it's useful first |
|---|---|---|
| `hello` | Hello | Opens an interaction |
| `thank_you` | Thank you | Closes an interaction politely |
| `please` | Please | Softens a request |
| `yes` | Yes | Fast confirmation |
| `no` | No | Fast confirmation |
| `help` | I need help | High-stakes, time-sensitive |
| `water` | Water / thirsty | Basic need |
| `toilet` | Where is the toilet? | Basic need |
| `wait` | Please wait | Manages expectations during a delay |
| `more` | More / again | General-purpose modifier |
| `stop` | Stop | Safety-relevant |
| `name` | My name is… (prompts fingerspelling/text entry) | Introductions |

## Rules for growing this list

- Every new label must be reviewed by a fluent ISL signer before being
  treated as "supported" in any demo claim.
- Keep labels short, snake_case, and unambiguous (`toilet`, not `bathroom`
  and `restroom` as two separate near-duplicate labels).
- A sign that differs mainly by facial expression, movement path, or
  non-manual markers (not just static hand shape) will not be recognized
  reliably by this MVP's static-landmark classifier — flag those as
  "needs sequence model" for the `/ml-model` upgrade path instead of
  enrolling them here and reporting false success.
