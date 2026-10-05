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
- (to fill in)
