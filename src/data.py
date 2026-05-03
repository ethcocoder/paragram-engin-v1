"""
data.py — Paradox Genesis: Stage-Partitioned Data Pipeline
===========================================================
Each training stage gets its own NON-OVERLAPPING, DETERMINISTIC slice of data.
This is critical for proper ML hygiene:

  Stage 1 — MONSTER     : STL-10 unlabeled (indices 0      → 49,999)   Large, diverse
  Stage 2 — FINE-TUNE   : STL-10 unlabeled (indices 50,000 → 74,999)   Fresh data, never seen in S1
  Stage 3 — RL MISTAKES : STL-10 unlabeled (indices 75,000 → 99,999)   Held-out challenge set
  Eval    — TEST        : STL-10 test split (800 images)                Never used in any training

All splits are FIXED (seed=42) so every run uses exactly the same images.
"""

import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from typing import Tuple, Optional

# ── Deterministic partition boundaries ──────────────────────────────────────
# STL-10 unlabeled has exactly 100,000 images.
STAGE_SPLITS = {
    1: (0,      50_000),   # Monster     — 50k images, broad visual grammar
    2: (50_000, 75_000),   # Fine-Tune   — 25k images, quality polish domain
    3: (75_000, 100_000),  # RL Mistakes — 25k held-out images, never seen before
}


def _make_transform(size: int = 256, augment: bool = True) -> transforms.Compose:
    if augment:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(size, scale=(0.8, 1.0)),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])


def get_dataloaders(
    batch_size:   int = 12,
    root:         str = "./data",
    num_workers:  int = 2,
    pin_memory:   bool = True,
    use_hd:       bool = True,
    sample_limit: Optional[int] = None,
    stage:        int = 1,           # ← NEW: controls which data partition is used
) -> Tuple[DataLoader, DataLoader]:
    """
    Returns (trainloader, testloader) for the given stage.

    Stage partitions (STL-10 unlabeled, 100k total):
      stage=1 → indices [0,      50,000)   Monster training set
      stage=2 → indices [50,000, 75,000)   Fine-tune set  (never seen in stage 1)
      stage=3 → indices [75,000, 100,000)  RL held-out set (never seen in stages 1-2)

    sample_limit caps within the stage's own partition, not across the full dataset.
    """
    size = 256 if use_hd else 32

    train_tf = _make_transform(size, augment=True)
    test_tf  = _make_transform(size, augment=False)

    if use_hd:
        # Full unlabeled set (100k) — we slice it per stage
        full_train = torchvision.datasets.STL10(
            root=root, split='unlabeled', download=True, transform=train_tf)
        testset = torchvision.datasets.STL10(
            root=root, split='test', download=True, transform=test_tf)

        # ── Stage partition (deterministic) ──────────────────────────────────
        lo, hi = STAGE_SPLITS.get(stage, (0, len(full_train)))
        # Use a fixed generator so the same images are always selected
        rng = torch.Generator()
        rng.manual_seed(42 + stage)   # stage-specific seed → reproducible & distinct
        perm = torch.randperm(hi - lo, generator=rng) + lo   # shuffle within partition

        if sample_limit and sample_limit < len(perm):
            perm = perm[:sample_limit]

        trainset = Subset(full_train, perm.tolist())

    else:
        # CIFAR-10 fallback (used if use_hd=False)
        trainset = torchvision.datasets.CIFAR10(
            root=root, train=True, download=True, transform=train_tf)
        testset = torchvision.datasets.CIFAR10(
            root=root, train=False, download=True, transform=test_tf)

        if sample_limit and sample_limit < len(trainset):
            rng = torch.Generator(); rng.manual_seed(42 + stage)
            idx = torch.randperm(len(trainset), generator=rng)[:sample_limit]
            trainset = Subset(trainset, idx.tolist())

    _persistent = num_workers > 0
    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=_persistent
    )
    testloader = DataLoader(
        testset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=_persistent
    )

    n_train = len(trainset)
    lo, hi  = STAGE_SPLITS.get(stage, (0, 0))
    print(f"[Data] Stage {stage} partition: indices [{lo:,} → {hi:,}] | "
          f"Using {n_train:,} images | Batch: {batch_size}")

    return trainloader, testloader
