# Learnings (semantic memory)

## Environment facts
- CPU-only box: 2 cores, ~2 GB RAM. No GPU. torch 2.7.1+cpu, numpy/pandas/pyarrow/tiktoken/rustbpe available.
- Real autoresearch data in place: climbmix shards 0,1,2 (train) + 6542 (val) at ~/.cache/autoresearch/data; BPE tokenizer (vocab 8192) trained and cached.
- `uv` shim script maps `uv run X` -> `python3 X`. Timeout wrapper: `timeout 300 ...`.
- Locked evaluator: evaluate_bpb(model, tok, B) runs EVAL_TOKENS/(B*2048) full 2048-token rows on val shard. Model forward MUST accept (idx, targets, reduction='none') and return per-token losses when reduction='none'.
- Eval cost scales with batch size: steps = 20971520/(B*2048). B=4 -> 2560 rows of 2048.

## What works
- (to fill in)

## What doesn't
- Eval pieces of 128 rows cost ~30 s each on this box; with 1280 rows the schedule needs
  ~300 s of pure eval — interleaved planner still lands at the wall. Next: shrink per-row
  eval cost or reserve more end-of-run eval time (raise PIECE_FRAC / lower MIN_TRAIN_FRAC).
- Verification run 2026-10-05T14:52Z timed out at 640/1280 eval rows (logged in results.tsv).
