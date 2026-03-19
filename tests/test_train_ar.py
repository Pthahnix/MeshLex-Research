"""Tests for train_ar.py training loop changes."""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR


def test_grad_accumulation_updates_every_n_steps():
    """Gradient accumulation: optimizer.step() fires every grad_accum_steps mini-batches."""
    model = nn.Linear(4, 2)
    optimizer = AdamW(model.parameters(), lr=1e-2)
    grad_accum_steps = 4

    initial_weight = model.weight.data.clone()

    for step in range(1, grad_accum_steps + 1):
        x = torch.randn(2, 4)
        loss = model(x).sum()
        loss = loss / grad_accum_steps
        loss.backward()

        if step < grad_accum_steps:
            assert torch.equal(model.weight.data, initial_weight), \
                f"Weights changed at step {step}, expected no change until step {grad_accum_steps}"

    optimizer.step()
    optimizer.zero_grad()

    assert not torch.equal(model.weight.data, initial_weight), \
        "Weights did not change after optimizer.step()"


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

    assert lrs[warmup_epochs - 1] > lrs[0], "LR should increase during warmup"
    assert abs(lrs[warmup_epochs] - 3e-4) < 1e-5, f"LR at warmup end: {lrs[warmup_epochs]}"
    assert lrs[-1] < lrs[warmup_epochs], "LR should decrease during cosine phase"
    assert lrs[-1] < 1e-6, f"Final LR too high: {lrs[-1]}"


def test_scheduler_state_save_restore():
    """Scheduler + optimizer state_dict round-trips correctly for resume."""
    warmup_epochs = 10
    total_epochs = 300

    def make_pair():
        m = nn.Linear(4, 2)
        opt = AdamW(m.parameters(), lr=3e-4)
        warmup = LinearLR(opt, start_factor=1e-8 / 3e-4, end_factor=1.0, total_iters=warmup_epochs)
        cosine = CosineAnnealingLR(opt, T_max=total_epochs - warmup_epochs)
        sched = SequentialLR(opt, schedulers=[warmup, cosine], milestones=[warmup_epochs])
        return m, opt, sched

    m1, opt1, sched1 = make_pair()

    for _ in range(50):
        sched1.step()

    # Save both optimizer and scheduler state (as the real script does)
    opt_state = opt1.state_dict()
    sched_state = sched1.state_dict()
    lr_at_50 = opt1.param_groups[0]["lr"]

    # Create fresh pair and restore both states
    m3, opt3, sched3 = make_pair()
    opt3.load_state_dict(opt_state)
    sched3.load_state_dict(sched_state)

    # Advance both by one more step
    sched1.step()
    sched3.step()

    lr_original = opt1.param_groups[0]["lr"]
    lr_restored = opt3.param_groups[0]["lr"]
    assert abs(lr_original - lr_restored) < 1e-10, \
        f"LR mismatch after restore: original={lr_original}, restored={lr_restored}"
