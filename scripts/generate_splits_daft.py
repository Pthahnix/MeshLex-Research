"""Generate train/test/unseen splits from metadata + verify via Daft.

Usage:
    python scripts/generate_splits_daft.py \
        --hf_repo Pthahnix/MeshLex-Patches \
        --holdout_count 100 --test_ratio 0.2 --seed 42
"""
import argparse
import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np
from huggingface_hub import HfApi, hf_hub_download

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)


def generate_splits(metadata, holdout_count=100, test_ratio=0.2, seed=42):
    rng = np.random.default_rng(seed)
    cat_to_meshes = {}
    for mesh_id, info in metadata.items():
        cat_to_meshes.setdefault(info["category"], []).append(mesh_id)

    all_cats = sorted(cat_to_meshes.keys())
    actual_holdout = min(holdout_count, len(all_cats) // 2)
    perm = rng.permutation(len(all_cats))
    unseen_cats = [all_cats[i] for i in perm[:actual_holdout]]
    seen_cats = [all_cats[i] for i in perm[actual_holdout:]]

    unseen = [m for c in unseen_cats for m in cat_to_meshes[c]]
    seen_meshes = [m for c in seen_cats for m in cat_to_meshes[c]]
    rng.shuffle(seen_meshes)
    n_test = int(len(seen_meshes) * test_ratio)

    return {
        "seen_train": sorted(seen_meshes[n_test:]),
        "seen_test": sorted(seen_meshes[:n_test]),
        "unseen": sorted(unseen),
        "unseen_categories": sorted(unseen_cats),
        "seen_categories": sorted(seen_cats),
        "split_seed": seed, "holdout_count": actual_holdout, "test_ratio": test_ratio,
    }


def compute_stats(metadata, splits):
    total_patches = sum(m["n_patches"] for m in metadata.values())
    return {
        "total_meshes": len(metadata),
        "total_patches": total_patches,
        "avg_patches_per_mesh": round(total_patches / max(len(metadata), 1), 1),
        "source_counts": dict(Counter(m["source"] for m in metadata.values())),
        "n_categories": len(set(m["category"] for m in metadata.values())),
        "split_sizes": {k: len(splits[k]) for k in ["seen_train", "seen_test", "unseen"]},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_repo", default="Pthahnix/MeshLex-Patches")
    parser.add_argument("--holdout_count", type=int, default=100)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--work_dir", default="/tmp/meshlex_splits")
    args = parser.parse_args()

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    hf_api = HfApi()

    # Download metadata JSONs
    log.info("Downloading metadata from HF...")
    meta_all = {}
    for fname in ["metadata_objaverse.json", "metadata_shapenet.json"]:
        try:
            path = hf_hub_download(repo_id=args.hf_repo, filename=fname,
                                   repo_type="dataset", local_dir=str(work_dir))
            with open(path) as f:
                meta_all.update(json.load(f))
        except Exception as e:
            log.warning(f"Could not download {fname}: {e}")

    log.info(f"Total meshes in metadata: {len(meta_all)}")

    # Verify row count via Daft (lazy, no full download)
    import daft
    from src.daft_utils import get_hf_io_config
    io_config = get_hf_io_config()
    try:
        df = daft.read_parquet(
            f"hf://datasets/{args.hf_repo}/**/*.parquet", io_config=io_config,
        )
        parquet_rows = df.count_rows()
        log.info(f"Parquet row count (patches): {parquet_rows}")
    except Exception as e:
        log.warning(f"Could not read Parquet from HF: {e}")

    splits = generate_splits(meta_all, args.holdout_count, args.test_ratio, args.seed)
    stats = compute_stats(meta_all, splits)
    log.info(f"Splits: {stats['split_sizes']}")

    for name, data in [("metadata.json", meta_all), ("splits.json", splits), ("stats.json", stats)]:
        with open(work_dir / name, "w") as f:
            json.dump(data, f, indent=2)

    for name in ["metadata.json", "splits.json", "stats.json"]:
        hf_api.upload_file(path_or_fileobj=str(work_dir / name),
                           path_in_repo=name, repo_id=args.hf_repo, repo_type="dataset")

    log.info("Splits generation complete!")


if __name__ == "__main__":
    main()
