# Program: Autoresearch Benchmark — Optimize `train.py`

## Goal
Achieve the lowest possible **validation loss (`val_bpb`)** by editing **only `train.py`**,
under a strict per-run time budget. `bpb` = bits-per-byte of the model's cross-entropy
on held-out bytes (lower is better). This is a self-contained, CPU-friendly proxy for
LLM-pretraining optimization loops: the model learns to predict raw UTF-8 bytes of text.

## The Loop (one iteration)
1. **READ CONTEXT** — read this file, `prepare.py` (locked), and `results.tsv` (history).
2. **HYPOTHESIZE & EDIT** — propose **one** specific change to `train.py`
   (architecture, optimizer, hyperparameters, batching, schedule…). Apply it.
3. **COMMIT** — `git add train.py && git commit -m "<experiment>"`.
4. **EXECUTE (budgeted)** — `timeout 300 uv run train.py > run.log 2>&1`
   (fallback if `uv` is missing: `timeout 300 python3 train.py > run.log 2>&1`).
   **5-minute hard budget.** If exceeded, kill immediately → status `timeout`.
5. **VERIFY & SCORE** — parse `run.log` for the line `val_bpb=<float>` and
   `duration_s=<float>`. Crash → status `crash`; no score → status `timeout`.
6. **DECIDE (Git logic)**
   - Improved (lower `val_bpb` than best so far): **KEEP**, advance branch, log row.
   - Not improved / crash / timeout: **DISCARD**, `git reset --hard HEAD~1`, log row.
7. **REPEAT** — start the next iteration immediately. Never ask permission.

## Non-Negotiable Rules
- 🔒 **Locked judge:** `prepare.py` is read-only (also enforced via file permissions).
  It owns data generation, tokenization-equivalent (byte encoding), splits, and the
  `bpb` metric. You may not modify it or any evaluation logic.
- 📄 **One-file focus:** only `train.py` is mutable. All creativity lives there.
- ⏱️ **Fixed budget:** each run must finish within 300 s wall-clock. Longer = failure.
  Plan small: this box is CPU-only (2 cores, ~2 GB RAM).
- ✨ **Simplicity bias:** prefer elegance over hacks. Tiny gain + 20 hacky lines → drop.
  Same score + less code → keep. Auto-reject rule: if `wc -l train.py` grows by >20
  lines while `val_bpb` improves <1%, reject.
- 🧠 **Memory:** every experiment logged in `results.tsv`; durable insights distilled
  into `learnings.md`; review both at the start of every 10th iteration.

## Scoring
`val_bpb` is computed by `prepare.evaluate(model)` on a fixed held-out byte sequence
(never seen in training). Training data and validation data are disjoint.
The baseline committed in `train.py` defines the initial best score.

## Notes for constrained environments
- No GPU, no internet needed after setup. Pure PyTorch-CPU is fine.
- `uv run` works as a shim when `uv` is absent (see `uv` wrapper script in repo root);
  otherwise use the documented fallback command.
