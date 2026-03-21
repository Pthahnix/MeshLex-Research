"""Train MeshLex RVQ-VAE on preprocessed patches.

Usage:
    # From NPZ directories (original):
    python scripts/train_rvq.py \
        --train_dirs data/patches/lvis_wide/seen_train \
        --val_dirs data/patches/lvis_wide/seen_test \
        --checkpoint_dir data/checkpoints/rvq_lvis \
        --epochs 200 --batch_size 256

    # From Parquet (full-scale):
    python scripts/train_rvq.py \
        --parquet_dir /data/pthahnix/MeshLex-Research/datasets/MeshLex-Patches/data \
        --splits_json data/splits.json \
        --checkpoint_dir data/checkpoints/rvq_full_pca \
        --epochs 100 --batch_size 1024
"""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset

from src.model_rvq import MeshLexRVQVAE
from src.patch_dataset import PatchGraphDataset
from src.trainer import Trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dirs", nargs="+", default=None,
                        help="NPZ patch directories for training")
    parser.add_argument("--val_dirs", nargs="+", default=None)
    parser.add_argument("--parquet_dir", type=str, default=None,
                        help="Parquet directory (alternative to --train_dirs)")
    parser.add_argument("--splits_json", type=str, default=None,
                        help="splits.json path (required with --parquet_dir)")
    parser.add_argument("--codebook_size", type=int, default=1024)
    parser.add_argument("--n_levels", type=int, default=3)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_kv_tokens", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--checkpoint_dir", type=str, default="data/checkpoints/rvq")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--nopca", action="store_true", help="Train on non-PCA-normalized vertices")
    parser.add_argument("--vq_method", choices=["simvq", "vanilla", "ema"], default="simvq",
                        help="VQ codebook method")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.parquet_dir:
        from src.patch_dataset import ParquetPatchDataset
        # Load splits
        splits_path = args.splits_json
        if splits_path is None:
            # Try to download from HF
            from src.parquet_loader import download_splits_json
            splits_path = "data/splits.json"
            download_splits_json(output_path=splits_path)

        with open(splits_path) as f:
            splits = json.load(f)

        train_mesh_ids = set(splits["seen_train"])
        train_dataset = ParquetPatchDataset(
            args.parquet_dir, use_nopca=args.nopca, split_mesh_ids=train_mesh_ids)
        print(f"Training patches (parquet): {len(train_dataset)}")

        val_dataset = None
        if "seen_test" in splits:
            val_mesh_ids = set(splits["seen_test"])
            val_dataset = ParquetPatchDataset(
                args.parquet_dir, use_nopca=args.nopca, split_mesh_ids=val_mesh_ids)
            print(f"Validation patches (parquet): {len(val_dataset)}")
    elif args.train_dirs:
        train_datasets = [PatchGraphDataset(d, use_nopca=args.nopca) for d in args.train_dirs]
        train_dataset = ConcatDataset(train_datasets)
        print(f"Training patches: {len(train_dataset)}")

        val_dataset = None
        if args.val_dirs:
            val_datasets = [PatchGraphDataset(d, use_nopca=args.nopca) for d in args.val_dirs]
            val_dataset = ConcatDataset(val_datasets)
            print(f"Validation patches: {len(val_dataset)}")
    else:
        raise ValueError("Must specify either --train_dirs or --parquet_dir")

    model = MeshLexRVQVAE(
        codebook_size=args.codebook_size,
        n_levels=args.n_levels,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_kv_tokens=args.num_kv_tokens,
        vq_method=args.vq_method,
    )

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"RVQ-VAE: {n_params:,} total, {n_trainable:,} trainable")
    print(f"Codebook: {args.n_levels} levels x K={args.codebook_size}")

    ckpt_data = None
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
        print(f"Resumed from {args.resume}")
        if not missing and not unexpected:
            ckpt_data = ckpt

    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        checkpoint_dir=args.checkpoint_dir,
        device=device,
        resume_checkpoint=ckpt_data,
    )

    # Save training config
    config = {
        "codebook_size": args.codebook_size,
        "n_levels": args.n_levels,
        "embed_dim": args.embed_dim,
        "hidden_dim": args.hidden_dim,
        "num_kv_tokens": args.num_kv_tokens,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "nopca": args.nopca,
        "vq_method": args.vq_method,
        "train_dirs": args.train_dirs,
        "val_dirs": args.val_dirs,
    }
    config_path = Path(args.checkpoint_dir) / "config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config saved to {config_path}")

    trainer.train()


if __name__ == "__main__":
    main()
