"""
Autoresearch pretraining script — CPU edition (baseline).
Single-file, time-budgeted. Usage: uv run train.py  (or: python3 train.py)

Trains a small decoder-only Transformer on climbmix BPE tokens and reports
val_bpb (bits-per-byte, lower is better) computed by the locked judge in prepare.py.
Only this file is mutable; prepare.py / program.md are read-only.
"""

import math
import os
import sys
import time
import itertools
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare import (MAX_SEQ_LEN, TIME_BUDGET, EVAL_TOKENS, Tokenizer,
                     make_dataloader)

torch.manual_seed(42)
torch.set_num_threads(2)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

@dataclass
class GPTConfig:
    vocab_size: int = 8192
    n_layer: int = 2
    n_head: int = 4
    n_embd: int = 128
    rope_base: float = 10000.0


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)

    def forward(self, x, cos, sin, causal_mask):
        B, T, C = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.n_head, self.head_dim).unbind(dim=2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))  # (B,H,T,hd)
        c, s = cos[:T].unsqueeze(0).unsqueeze(0), sin[:T].unsqueeze(0).unsqueeze(0)
        hd = self.head_dim

        def rope(t):
            t1, t2 = t[..., : hd // 2], t[..., hd // 2:]
            return torch.cat([t1 * c - t2 * s, t2 * c + t1 * s], dim=-1)

        q, k = rope(q), rope(k)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=causal_mask[:T, :T])
        return self.proj(y.transpose(1, 2).reshape(B, T, C))


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=False)

    def forward(self, x):
        return self.c_proj(F.relu(self.c_fc(x)).square())


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn = CausalSelfAttention(cfg)
        self.mlp = MLP(cfg)

    def forward(self, x, cos, sin, causal_mask):
        x = x + self.attn(norm(x), cos, sin, causal_mask)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    """Forward signature matches the locked evaluator: model(x, y, reduction='none')."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        head_dim = cfg.n_embd // cfg.n_head
        pos = torch.arange(MAX_SEQ_LEN, dtype=torch.float32)
        inv = 1.0 / (cfg.rope_base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        freqs = torch.outer(pos, inv)
        self.register_buffer("rot_cos", freqs.cos(), persistent=False)
        self.register_buffer("rot_sin", freqs.sin(), persistent=False)
        self.register_buffer("causal_mask",
                             torch.triu(torch.ones(MAX_SEQ_LEN, MAX_SEQ_LEN, dtype=torch.bool),
                                        diagonal=1), persistent=False)

    def forward(self, idx, targets=None, reduction="mean", chunk=8):
        # Chunk along batch dim so peak memory stays bounded (lm_head logits dominate:
        # B*T*V floats; on this 2 GB CPU box keep the chunk <= a few hundred MB).
        if idx.size(0) <= chunk:
            return self._forward(idx, targets, reduction)
        outs = [self._forward(idx[i:i + chunk],
                              None if targets is None else targets[i:i + chunk], reduction)
                for i in range(0, idx.size(0), chunk)]
        if reduction == "none":
            return torch.cat(outs, 0)
        return sum(o * o.new_tensor(s) for o, s in zip(outs, [idx[i:i + chunk].numel()
                        for i in range(0, idx.size(0), chunk)])) / idx.numel()

    def _forward(self, idx, targets, reduction):
        T = idx.size(1)
        x = self.wte(idx)
        for block in self.blocks:
            x = block(x, self.rot_cos[:T], self.rot_sin[:T], self.causal_mask)
        logits = self.lm_head(norm(x))
        if targets is None:
            return logits
        return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                               ignore_index=-1, reduction=reduction)

# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly, no CLI flags needed)
# ---------------------------------------------------------------------------

N_LAYER = 2
N_HEAD = 4
N_EMBD = 96
SEQ_LEN = 256           # training context length (eval rows are always MAX_SEQ_LEN)
BATCH = 4               # rows per training step

LR = 3e-3               # AdamW learning rate
WEIGHT_DECAY = 0.1
WARMDOWN = 0.1          # linear decay over final 10% of the training slice

EVAL_BATCH = 8          # rows per eval dataloader step
SELF_CHUNK = 4          # rows per eval forward pass (4*2048*8192*4B = 0.27 GB logits)
EVAL_ROWS = EVAL_TOKENS // (EVAL_BATCH * MAX_SEQ_LEN)   # 1280 rows total
# Interleaved train/eval schedule: the eval cost is unknown up front, so we
# alternate train slices and eval pieces and re-plan after every piece using
# measured timings. Guarantees the run finishes inside the 300 s hard budget.
TARGET_END = 285.0      # aim to be fully done by here
PIECE_FRAC = 0.1        # first eval piece = 10% of the rows; then re-plan
MIN_TRAIN_FRAC = 0.3    # always keep at least 30% of the budget for training


def pick_switch(el, est):
    """Next switch-to-eval time given elapsed el and estimated remaining eval est."""
    lo = max(el + est + 3.0, MIN_TRAIN_FRAC * TIME_BUDGET)
    hi = max(el + 1.0, TARGET_END - el - est)
    return min(lo, hi)

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

t_start = time.time()

tokenizer = Tokenizer.from_directory()
vocab_size = tokenizer.get_vocab_size()
print(f"Vocab size: {vocab_size:,}")

cfg = GPTConfig(vocab_size=vocab_size, n_layer=N_LAYER, n_head=N_HEAD, n_embd=N_EMBD)
print(f"Model config: {asdict(cfg)}")
model = GPT(cfg)
num_params = sum(p.numel() for p in model.parameters())
print(f"num_params_M: {num_params / 1e6:.2f}")

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
train_loader = make_dataloader(tokenizer, BATCH, SEQ_LEN, "train")

# ---------------------------------------------------------------------------
# Main loop: alternate training slices and eval pieces, re-planning after
# each piece from measured timings. Eval replicates the locked bpb metric
# exactly (same loader stream & order), but shares ONE val dataloader across
# pieces — packing cost is amortized instead of repeated per piece.
# ---------------------------------------------------------------------------

step = 0
tokens_seen = 0
smooth_loss = 0.0
eval_batches = None
piece_times = []

t_eval0 = time.time()
from prepare import get_token_bytes as _gtb
token_bytes = _gtb(device="cpu")

total_nats = 0.0
total_bytes = 0
rows_done = 0
piece_rows = max(EVAL_BATCH, int(EVAL_ROWS * PIECE_FRAC) // EVAL_BATCH * EVAL_BATCH)
est_remaining = [TARGET_END * 0.5]   # rough initial guess; refined after first piece
switch_at = pick_switch(time.time() - t_start, est_remaining[0])
print(f"eval plan: pieces of {piece_rows} rows, first switch at {switch_at:.0f}s", flush=True)

while True:
    # ---- train until switch point ----
    while time.time() - t_start < switch_at:
        # progress toward the projected end (TARGET_END); decays LR over final part
        proj = min(TARGET_END, time.time() - t_start + est_remaining[0])
        frac = min((time.time() - t_start) / max(proj, 1.0), 1.0)
        lrm = 1.0 if frac <= 1.0 - WARMDOWN else max(0.05, (1.0 - frac) / WARMDOWN)
        for g in optimizer.param_groups:
            g["lr"] = LR * lrm
        x, y, epoch = next(train_loader)
        loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        tokens_seen += x.numel()
        step += 1
        smooth_loss = 0.9 * smooth_loss + 0.1 * loss.item()
        if step % 50 == 0:
            el = time.time() - t_start
            print(f"step {step:5d} | loss ~{smooth_loss / (1 - 0.9 ** step):.4f} | "
                  f"tok {tokens_seen / 1e6:.2f}M | lr {optimizer.param_groups[0]['lr']:.2e} | "
                  f"elapsed {el:.0f}s", flush=True)

    # ---- one eval piece (forward batched; packing amortized across rows) ----
    with torch.no_grad():
        model.eval()
        if eval_batches is None:
            eval_batches = make_dataloader(tokenizer, EVAL_BATCH, MAX_SEQ_LEN, "val")
        t_piece = time.time()
        target = rows_done + piece_rows
        while rows_done < target:
            xb, yb, epoch = next(eval_batches)
            nb = xb.size(0)
            take = min(nb, target - rows_done)
            xk, yk = xb[:take], yb[:take]
            loss_flat = model(xk, yk, reduction="none", chunk=SELF_CHUNK).view(-1)
            nbytes = token_bytes[yk.view(-1)]                     # (take*T,)
            mask = nbytes > 0
            total_nats += (loss_flat * mask).sum().item()
            total_bytes += nbytes.sum().item()
            rows_done += take
        pt = time.time() - t_piece
        piece_times.append(pt)
        el = time.time() - t_start
        remaining = math.ceil((EVAL_ROWS - rows_done) / piece_rows)
        est = sum(piece_times[-2:]) / len(piece_times[-2:]) * remaining
        est_remaining[0] = est
        # re-plan: keep >=MIN_TRAIN_FRAC of budget training, leave room to finish eval
        switch_at = pick_switch(el, est)
        model.train()
        print(f"eval piece done ({rows_done}/{EVAL_ROWS} rows, {pt:.0f}s) | "
              f"next switch at {switch_at:.0f}s", flush=True)
    if rows_done >= EVAL_ROWS:
        break

val_bpb = total_nats / (math.log(2.0) * total_bytes)

t_end = time.time()
print("---")
print(f"val_bpb:        {val_bpb:.6f}")
print(f"steps:          {step}")
print(f"train_tokens_M: {tokens_seen / 1e6:.2f}")
print(f"eval_rows:      {rows_done}/{EVAL_ROWS}")
print(f"eval_s:         {t_end - t_eval0:.1f}")
print(f"duration_s:     {t_end - t_start:.1f}")
