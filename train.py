"""
Autoresearch pretraining script — CPU edition (baseline).
Single-file, time-budgeted. Usage: uv run train.py  (or: python3 train.py)

Trains a small decoder-only Transformer on climbmix BPE tokens and reports
val_bpb (bits-per-byte, lower is better) computed by the locked judge in prepare.py.
Only this file is mutable; prepare.py / program.md are read-only.
"""

import os
import sys
import time
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare import MAX_SEQ_LEN, TIME_BUDGET, Tokenizer, make_dataloader, evaluate_bpb

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

    def forward(self, idx, targets=None, reduction="mean"):
        x = self.wte(idx)
        for block in self.blocks:
            x = block(x, self.rot_cos, self.rot_sin, self.causal_mask)
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
N_EMBD = 128
SEQ_LEN = 256           # training context length (eval rows are always MAX_SEQ_LEN)
BATCH = 16              # rows per training step

LR = 3e-3               # AdamW learning rate
WEIGHT_DECAY = 0.1
WARMDOWN = 0.1          # linear decay over final 10% of the token budget

TRAIN_TOKENS_CAP = 500_000     # ~120s of training at ~4.7 ktok/s on this box
EVAL_RESERVE_S = 170           # stop training by TIME_BUDGET - reserve
EVAL_BATCH = 16                # batch size passed to the locked evaluator

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
# Training loop (budget-aware: TRAIN_TOKENS_CAP or time guard, whichever first)
# ---------------------------------------------------------------------------

step = 0
tokens_seen = 0
smooth_loss = 0.0
while tokens_seen < TRAIN_TOKENS_CAP:
    if time.time() - t_start > TIME_BUDGET - EVAL_RESERVE_S:
        print("time guard tripped, stopping training early")
        break
    frac = min(tokens_seen / TRAIN_TOKENS_CAP, 1.0)
    for g in optimizer.param_groups:
        g["lr"] = LR * max(0.0, (1.0 - frac) / WARMDOWN if frac > 1.0 - WARMDOWN else 1.0)
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

# ---------------------------------------------------------------------------
# Evaluation (locked metric)
# ---------------------------------------------------------------------------

t_eval0 = time.time()
model.eval()
val_bpb = evaluate_bpb(model, tokenizer, EVAL_BATCH)

t_end = time.time()
print("---")
print(f"val_bpb:        {val_bpb:.6f}")
print(f"steps:          {step}")
print(f"train_tokens_M: {tokens_seen / 1e6:.2f}")
print(f"eval_s:         {t_end - t_eval0:.1f}")
print(f"duration_s:     {t_end - t_start:.1f}")
