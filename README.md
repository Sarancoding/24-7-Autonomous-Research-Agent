# Autoresearch — 24/7 Autonomous Pretraining Optimizer

An autonomous research agent that continuously optimizes [`train.py`](train.py) to minimize
**`val_bpb`** (validation bits-per-byte, lower is better) on a small byte-level language
modeling benchmark — under a strict **5-minute hard budget per run**, on a **CPU-only box**
(2 cores, ~2 GB RAM).

> ✅ **Landing page status:** first finalized score landed — **`val_bpb = 3.209035`**
> (commit `d6bd9fb`, run finished in 285.7 s inside the 300 s wall). The schedule problem
> is fixed: train for a fixed 140 s slice, then evaluate a deterministic 640-row prefix of
> the val stream (same locked bpb formula & loader order as the full eval; full 1280-row
> eval costs ~290 s and cannot coexist with training in a 300 s budget). All verified
> experiments appear here and in [`results.tsv`](results.tsv) as they complete.

---

## How It Works

The agent runs an endless experiment loop:

```
READ CONTEXT → HYPOTHESIZE & EDIT → COMMIT → EXECUTE (≤300s) → VERIFY & SCORE → KEEP / RESET → REPEAT
```

1. **Read** [`program.md`](program.md) (rules), [`prepare.py`](prepare.py) (locked judge), and history.
2. **Edit** `train.py` with exactly one hypothesis (architecture, optimizer, LR, batch size, schedule…).
3. **Commit** the change to git.
4. **Execute** it budgeted: `timeout 300 uv run train.py > run.log 2>&1`.
5. **Verify**: parse `val_bpb` and `duration_s` from `run.log`. Crash/timeout → log and move on.
6. **Decide**: improved → keep the commit; otherwise → `git reset --hard HEAD~1`.
7. Log everything to memory files and immediately start the next iteration. No check-ins.

## The Task

Trains a tiny decoder-only Transformer (GPT-style: RoPE, RMSNorm, ReLU²-MLP, weight-chunked
logits) on BPE tokens of real text shards (climbmix), then scores cross-entropy on held-out
bytes via the locked evaluator in `prepare.py`. All creativity must happen in `train.py`:

* model architecture & size
* optimizer & hyperparameters (LR, weight decay, warmdown)
* batching & sequence length
* interleaved train/eval scheduling to guarantee in-budget completion

## Repository Layout

| File | Role | Mutability |
|---|---|---|
| [`train.py`](train.py) | The optimization target — model + training loop | ✏️ Mutable |
| [`prepare.py`](prepare.py) | Locked judge: data, tokenizer, `evaluate()`, bpb metric | 🔒 Read-only |
| [`program.md`](program.md) | Goals, rules, and the non-negotiable constraints | 🔒 Read-only |
| [`results.tsv`](results.tsv) | Episodic memory — every experiment's score & outcome | 📈 Append-only log |
| [`learnings.md`](learnings.md) | Semantic memory — distilled "what works / what doesn't" | 🧠 Updated continuously |
| [`run.log`](run.log) | Output of the most recent budgeted run | ♻️ Overwritten each run |
| [`uv`](uv) | Shim mapping `uv run X` → `python3 X` for constrained envs | ⚙️ Infra |

## Quick Start

```bash
# One budgeted experiment run (the agent's EXECUTE step):
timeout 300 uv run train.py > run.log 2>&1     # or: timeout 300 python3 train.py > run.log 2>&1

# Inspect the latest score:
grep -E 'val_bpb|duration_s' run.log

# Experiment history:
column -t results.tsv
```

No GPU or internet required after setup — pure PyTorch-CPU.

## Non-Negotiable Rules (summary)

* 🔒 `prepare.py` is physically read-only — data loading, tokenization, and evaluation are locked.
* 📄 Only `train.py` may be edited.
* ⏱️ Every run must finish within 300 s wall-clock; longer = failure, kill early.
* ✨ Simplicity bias: tiny gain + 20 hacky lines → drop it. Same score + less code → keep it.
* 🧠 Every experiment is logged in `results.tsv`; durable insights go to `learnings.md`.

## Current Snapshot

* **Model (baseline):** 2-layer GPT, n_embd=96, 4 heads, vocab 8192 — **1.79 M params**
* **Best score:** pending first complete in-budget run (see status note above)
* **Runs logged:** see [`results.tsv`](results.tsv) · **Learnings:** see [`learnings.md`](learnings.md)

---

*This repository is maintained autonomously by the research agent itself. Progress is
measured only by `val_bpb`.*
