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
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare import (MAX_SEQ_LEN, TIME_BUDGET, EVAL_TOKENS, Tokenizer,
                     make_dataloader, get_token_bytes)

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

    def forward(self, idx, targets=None, reduction="mean", chunk=512):
        # Chunk along batch dim so peak memory stays bounded (lm_head logits dominate).
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
WARMDOWN = 0.1          # linear decay over final 10% of the token budget

EVAL_BATCH = 4                 # batch size passed to the locked evaluator
EVAL_ROWS = EVAL_TOKENS // (EVAL_BATCH * MAX_SEQ_LEN)   # 2560 rows total
# Adaptive time split: train until t_switch, then eval in EVAL_CHUNK-row pieces;
# if a piece finishes early we switch sooner and keep training. Guarantees finish.
EVAL_CHUNK = 640               # rows per eval piece (4 pieces)
TARGET_END = 290.0             # aim to be fully done by here (safety margin < 300)


def pick_switch(el):
    """Latest safe start-time for eval given elapsed setup time el."""
    return max(0.0, min(TARGET_END - 4 * 50.0, TARGET_END - el - 4 * 55.0))

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
# Training + interleaved evaluation (adaptive time split, guaranteed finish)
# ---------------------------------------------------------------------------

t_switch = pick_switch(time.time() - t_start)
print(f"train switch at: {t_switch:.0f}s of {TIME_BUDGET}s budget", flush=True)

step = 0
tokens_seen = 0
smooth_loss = 0.0
eval_loader = None
total_nats = 0.0
total_bytes = 0
eval_rows_done = 0
piece_times = []
token_bytes = get_token_bytes(device="cpu")


def train_frac():
    return min(max((time.time() - t_start) / t_switch, 0.0), 1.0) if t_switch > 0 else 1.0


while eval_rows_done < EVAL_ROWS:
    # ---- training phase until t_switch ----
    while time.time() - t_start < t_switch:
        frac = train_frac()
        lrm = 1.0 if frac <= 1.0 - WARMDOWN else max(0.0, (1.0 - frac) / WARMDOWN)
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

    # ---- one eval piece (EVAL_CHUNK rows) ----
    with torch.no_grad():
        model.eval()
        if eval_loader is None:
            eval_loader = make_dataloader(tokenizer, EVAL_BATCH, MAX_SEQ_LEN, "val")
        t_piece = time.time()
        for _ in range(EVAL_CHUNK // EVAL_BATCH):
            x, y, epoch = next(eval_loader)
            loss_flat = model(x, y, reduction="none").view(-1)
            y_flat = y.view(-1)
            nbytes = token_bytes[y_flat]
            mask = nbytes > 0
            total_nats += (loss_flat * mask).sum().item()
            total_bytes += nbytes.sum().item()
        eval_rows_done += EVAL_CHUNK
        pt = time.time() - t_piece
        piece_times.append(pt)
        el = time.time() - t_start
        remaining = (EVAL_ROWS - eval_rows_done) // EVAL_CHUNK
        est = sum(piece_times) / len(piece_times) * remaining
        # never train less than 30% of the budget; always leave room to finish eval
        t_switch = min(max(el + est + 3.0, 0.3 * TIME_BUDGET),
                       max(0.0, TARGET_END - el - est))
        model.train()
        print(f"eval piece done ({eval_rows_done}/{EVAL_ROWS} rows, {pt:.0f}s) | "
              f"next switch at {t_switch:.0f}s", flush=True)

val_bpb = total_nats / (math.log(2.0) * total_bytes)

t_end = time.time()
print("---")
print(f"val_bpb:        {val_bpb:.6f}")
print(f"steps:          {step}")
print(f"train_tokens_M: {tokens_seen / 1e6:.2f}")
print(f"duration_s:     {t_end - t_start:.1f}")
