# AR Loss Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix AR loss from 5.41 to ≤ 4.0 by right-sizing PatchGPT and fixing the training recipe.

**Architecture:** Only `scripts/train_ar.py` changes. The model (`src/ar_model.py`) already accepts all dims as constructor args — we just pass smaller defaults. Training loop gets gradient accumulation, LR warmup via SequentialLR, and scheduler state in checkpoints.

**Tech Stack:** PyTorch (AdamW, CosineAnnealingLR, LinearLR, SequentialLR), existing PatchGPT + MeshSequenceDataset.

**Spec:** `docs/superpowers/specs/2026-03-19-ar-loss-fix-design.md`

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `scripts/train_ar.py` | Modify | CLI defaults, grad accum loop, SequentialLR, scheduler checkpoint |
| `tests/test_train_ar.py` | Create | Unit tests for training loop changes |

No other files are touched.

---

### Task 1: Add test for gradient accumulation logic

**Files:**
- Create: `tests/test_train_ar.py`

- [ ] **Step 1: Write the failing test**

```python
"""Tests for train_ar.py training loop changes."""

import torch
import torch.nn as nn
from torch.optim import AdamW


def test_grad_accumulation_updates_every_n_steps():
    """Gradient accumulation: optimizer.step() fires every grad_accum_steps mini-batches."""
    # Simple linear model to track weight changes
    model = nn.Linear(4, 2)
    optimizer = AdamW(model.parameters(), lr=1e-2)
    grad_accum_steps = 4

    initial_weight = model.weight.data.clone()

    for step in range(1, grad_accum_steps + 1):
        x = torch.randn(2, 4)
        loss = model(x).sum()
        loss = loss / grad_accum_steps  # scale loss
        loss.backward()

        if step < grad_accum_steps:
            # Weights should NOT have changed yet (no optimizer.step)
            assert torch.equal(model.weight.data, initial_weight), \
                f"Weights changed at step {step}, expected no change until step {grad_accum_steps}"

    # Now step
    optimizer.step()
    optimizer.zero_grad()

    # Weights MUST have changed
    assert not torch.equal(model.weight.data, initial_weight), \
        "Weights did not change after optimizer.step()"
```

- [ ] **Step 2: Run test to verify it passes (this is a logic validation test)**

Run: `python -m pytest tests/test_train_ar.py::test_grad_accumulation_updates_every_n_steps -v`
Expected: PASS (this test validates the accumulation pattern we'll implement)

- [ ] **Step 3: Commit**

```bash
git add tests/test_train_ar.py
git commit -m "test: add gradient accumulation logic test for train_ar"
```

---

### Task 2: Add test for SequentialLR warmup + cosine schedule

**Files:**
- Modify: `tests/test_train_ar.py`

- [ ] **Step 1: Write the test**

```python
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR


def test_sequential_lr_warmup_then_cosine():
    """SequentialLR: linear warmup for N epochs, then cosine decay."""
    model = nn.Linear(4, 2)
    optimizer = AdamW(model.parameters(), lr=3e-4)
    warmup_epochs = 10
    total_epochs = 300

    warmup_scheduler = LinearLR(
        optimizer, start_factor=1e-8 / 3e-4, end_factor=1.0, total_iters=warmup_epochs
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer, T_max=total_epochs - warmup_epochs
    )
    scheduler = SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs],
    )

    lrs = []
    for epoch in range(total_epochs):
        lrs.append(optimizer.param_groups[0]["lr"])
        scheduler.step()

    # LR should increase during warmup
    assert lrs[warmup_epochs - 1] > lrs[0], "LR should increase during warmup"
    # LR should be near peak at end of warmup
    assert abs(lrs[warmup_epochs] - 3e-4) < 1e-5, f"LR at warmup end: {lrs[warmup_epochs]}"
    # LR should decrease during cosine phase
    assert lrs[-1] < lrs[warmup_epochs], "LR should decrease during cosine phase"
    # LR should be near zero at end
    assert lrs[-1] < 1e-6, f"Final LR too high: {lrs[-1]}"
```

- [ ] **Step 2: Run test**

Run: `python -m pytest tests/test_train_ar.py::test_sequential_lr_warmup_then_cosine -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_train_ar.py
git commit -m "test: add SequentialLR warmup+cosine schedule test"
```

---

### Task 3: Add test for scheduler state save/restore

**Files:**
- Modify: `tests/test_train_ar.py`

- [ ] **Step 1: Write the test**

```python
def test_scheduler_state_save_restore():
    """Scheduler state_dict round-trips correctly for resume."""
    warmup_epochs = 10
    total_epochs = 300

    def make_pair():
        m = nn.Linear(4, 2)
        opt = AdamW(m.parameters(), lr=3e-4)
        warmup = LinearLR(opt, start_factor=1e-8 / 3e-4, end_factor=1.0, total_iters=warmup_epochs)
        cosine = CosineAnnealingLR(opt, T_max=total_epochs - warmup_epochs)
        sched = SequentialLR(opt, schedulers=[warmup, cosine], milestones=[warmup_epochs])
        return opt, sched

    opt1, sched1 = make_pair()
    opt2, sched2 = make_pair()

    # Advance both to epoch 50
    for _ in range(50):
        sched1.step()
        sched2.step()

    # Save state from sched1
    state = sched1.state_dict()
    lr_at_50 = opt1.param_groups[0]["lr"]

    # Verify both are at the same LR
    assert abs(opt1.param_groups[0]["lr"] - opt2.param_groups[0]["lr"]) < 1e-10

    # Create a fresh scheduler and restore from saved state
    opt3, sched3 = make_pair()
    sched3.load_state_dict(state)

    # Advance original and restored by one more step
    sched1.step()
    sched3.step()

    # LRs must match after restore + step
    lr_original = opt1.param_groups[0]["lr"]
    lr_restored = opt3.param_groups[0]["lr"]
    assert abs(lr_original - lr_restored) < 1e-10, \
        f"LR mismatch after restore: original={lr_original}, restored={lr_restored}"
```

- [ ] **Step 2: Run test**

Run: `python -m pytest tests/test_train_ar.py::test_scheduler_state_save_restore -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_train_ar.py
git commit -m "test: add scheduler state save/restore test"
```

---

### Task 4: Update CLI defaults in train_ar.py

**Files:**
- Modify: `scripts/train_ar.py:1-39`

- [ ] **Step 1: Update imports and CLI argument defaults**

Replace the imports and argparse section. Changes:
- Add `LinearLR, SequentialLR` imports
- Change `--d_model` default from 768 → 512
- Change `--n_heads` default from 12 → 8
- Change `--n_layers` default from 12 → 6
- Change `--epochs` default from 100 → 300
- Add `--grad_accum_steps` (default 8)
- Add `--warmup_epochs` (default 10)

```python
"""Train AR generation model on patch token sequences.

Usage:
    python scripts/train_ar.py \
        --sequence_dir data/sequences/lvis_wide \
        --checkpoint_dir data/checkpoints/ar_rvq \
        --mode rvq --epochs 300 --batch_size 4
"""
import argparse
import json
import time
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from pathlib import Path
import gc

from src.ar_model import PatchGPT
from src.patch_dataset import MeshSequenceDataset
from src.patch_sequence import compute_vocab_size


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence_dir", required=True)
    parser.add_argument("--mode", choices=["simvq", "rvq"], default="rvq")
    parser.add_argument("--checkpoint_dir", default="data/checkpoints/ar")
    parser.add_argument("--codebook_size", type=int, default=1024,
                        help="Codebook K (1024 for RVQ, 4096 for SimVQ)")
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--grad_accum_steps", type=int, default=8,
                        help="Gradient accumulation steps (effective batch = batch_size * this)")
    parser.add_argument("--warmup_epochs", type=int, default=10,
                        help="Linear LR warmup epochs before cosine decay")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()
```

- [ ] **Step 2: Verify the file parses**

Run: `python -c "import ast; ast.parse(open('scripts/train_ar.py').read()); print('OK')"`
Expected: OK

- [ ] **Step 3: Commit**

```bash
git add scripts/train_ar.py
git commit -m "feat(train_ar): update CLI defaults for smaller model + add grad_accum/warmup args"
```

---

### Task 5: Replace scheduler and add scheduler state to checkpoints

**Files:**
- Modify: `scripts/train_ar.py:63-82`

- [ ] **Step 1: Replace CosineAnnealingLR with SequentialLR and update resume logic**

Replace the scheduler creation and resume block. Changes:
- Create `SequentialLR(LinearLR → CosineAnnealingLR)` instead of bare `CosineAnnealingLR`
- Restore `scheduler.state_dict()` on resume

```python
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # LR schedule: linear warmup → cosine annealing
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=1e-8 / args.lr,
        end_factor=1.0,
        total_iters=args.warmup_epochs,
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer, T_max=args.epochs - args.warmup_epochs
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[args.warmup_epochs],
    )

    start_epoch = 0
    history = []
    config = {
        "vocab_size": vocab_size,
        "d_model": args.d_model,
        "n_heads": args.n_heads,
        "n_layers": args.n_layers,
        "max_seq_len": args.max_seq_len,
    }

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        history = ckpt.get("history", [])
        print(f"Resumed from epoch {start_epoch}")
```

- [ ] **Step 2: Verify the file parses**

Run: `python -c "import ast; ast.parse(open('scripts/train_ar.py').read()); print('OK')"`
Expected: OK

- [ ] **Step 3: Commit**

```bash
git add scripts/train_ar.py
git commit -m "feat(train_ar): replace CosineAnnealingLR with SequentialLR warmup+cosine"
```

---

### Task 6: Implement gradient accumulation in training loop + scheduler in checkpoints

**Files:**
- Modify: `scripts/train_ar.py:84-151`

- [ ] **Step 1: Rewrite the training loop with gradient accumulation**

Replace the epoch loop. Changes:
- Scale loss by `1/grad_accum_steps`
- Only call `optimizer.step()` + `optimizer.zero_grad()` + `clip_grad_norm_` every `grad_accum_steps` batches (or at end of epoch)
- Move `optimizer.zero_grad()` to before the inner loop start
- Add `scheduler_state_dict` to all checkpoint saves
- Print effective batch size and grad_accum info at start

```python
    print(f"Grad accumulation: {args.grad_accum_steps} steps "
          f"(effective batch = {args.batch_size * args.grad_accum_steps})")
    print(f"LR schedule: {args.warmup_epochs} warmup epochs → cosine to epoch {args.epochs}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = 0
        n_batches = 0
        t0 = time.time()
        optimizer.zero_grad()

        for batch_idx, (input_ids, target_ids) in enumerate(loader):
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)

            logits = model(input_ids)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                target_ids.reshape(-1),
                ignore_index=-100,
            )
            loss = loss / args.grad_accum_steps
            loss.backward()

            total_loss += loss.item() * args.grad_accum_steps  # unscaled for logging
            n_batches += 1

            if (batch_idx + 1) % args.grad_accum_steps == 0 or (batch_idx + 1) == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)
        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]
        torch.cuda.empty_cache()

        metrics = {"epoch": epoch, "loss": avg_loss, "lr": lr_now, "time_sec": elapsed}
        history.append(metrics)
        print(f"Epoch {epoch:03d} | loss {avg_loss:.4f} | lr {lr_now:.2e} | {elapsed:.1f}s")

        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            ckpt_path = ckpt_dir / f"checkpoint_epoch{epoch:03d}.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "history": history,
                "config": config,
            }, ckpt_path)
            # Keep only latest 3 checkpoints
            old_ckpts = sorted(ckpt_dir.glob("checkpoint_epoch*.pt"))[:-3]
            for old in old_ckpts:
                old.unlink()

        gc.collect()

    # Final checkpoint
    torch.save({
        "epoch": args.epochs - 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "history": history,
        "config": config,
    }, ckpt_dir / "checkpoint_final.pt")

    with open(ckpt_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"Training complete. Final checkpoint saved to {ckpt_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the file parses**

Run: `python -c "import ast; ast.parse(open('scripts/train_ar.py').read()); print('OK')"`
Expected: OK

- [ ] **Step 3: Run all tests**

Run: `python -m pytest tests/test_train_ar.py tests/test_ar_model.py -v`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add scripts/train_ar.py
git commit -m "feat(train_ar): implement gradient accumulation + scheduler checkpoints"
```

---

### Task 7: Smoke test — verify the full script runs

**Files:**
- No file changes

- [ ] **Step 1: Dry-run the training script for 2 epochs**

This verifies the full pipeline (data loading → model creation → grad accum → scheduler → checkpoint) works end-to-end with the new defaults.

Run:
```bash
python scripts/train_ar.py \
    --sequence_dir data/sequences/lvis_wide \
    --checkpoint_dir data/checkpoints/ar_v2_test \
    --mode rvq \
    --epochs 2 \
    --batch_size 4 \
    --grad_accum_steps 2 \
    --warmup_epochs 1
```

Expected output (approximate):
```
Sequences: 4674
Vocab size: 1856 (codebook K=1024)
PatchGPT: 20.4M params
Grad accumulation: 2 steps (effective batch = 8)
LR schedule: 1 warmup epochs → cosine to epoch 2
Epoch 000 | loss X.XXXX | lr X.XXe-XX | XX.Xs
Epoch 001 | loss X.XXXX | lr X.XXe-XX | XX.Xs
Training complete.
```

Verify:
- Param count shows ~20M (not 87M)
- No errors or crashes
- Checkpoint saved

- [ ] **Step 2: Clean up test checkpoint**

```bash
rm -rf data/checkpoints/ar_v2_test
```

- [ ] **Step 3: Final commit**

```bash
git add -A
git commit -m "feat: AR loss fix complete — model right-sized + training recipe updated"
```
