"""Pre-compute filtered Arrow datasets from Parquet files for each split.

Runs once. Produces:
  /data/pthahnix/MeshLex-Research/datasets/MeshLex-Patches/splits/seen_train/
  /data/pthahnix/MeshLex-Research/datasets/MeshLex-Patches/splits/seen_test/
  /data/pthahnix/MeshLex-Research/datasets/MeshLex-Patches/splits/unseen/

Each is a HF `datasets` directory that can be loaded instantly with:
  ds = datasets.load_from_disk(path)

Usage:
    PYTHONPATH=. python scripts/prepare_parquet_splits.py
"""
import json
import time
from pathlib import Path
from datasets import load_dataset


def main():
    parquet_dir = "/data/pthahnix/MeshLex-Research/datasets/MeshLex-Patches/data"
    splits_path = "data/splits.json"
    output_base = "/data/pthahnix/MeshLex-Research/datasets/MeshLex-Patches/splits"

    with open(splits_path) as f:
        splits = json.load(f)

    parquet_files = sorted(str(f) for f in Path(parquet_dir).glob("*.parquet"))
    print(f"Loading {len(parquet_files)} parquet files...")
    t0 = time.time()
    ds = load_dataset("parquet", data_files=parquet_files, split="train")
    print(f"Loaded {len(ds)} rows in {time.time()-t0:.1f}s")

    for split_name in ["seen_train", "seen_test", "unseen"]:
        split_ids = set(splits[split_name])
        out_dir = Path(output_base) / split_name

        if out_dir.exists() and (out_dir / "dataset_info.json").exists():
            print(f"  {split_name}: already exists, skipping")
            continue

        print(f"  Filtering {split_name} ({len(split_ids)} meshes)...")
        t1 = time.time()
        filtered = ds.filter(
            lambda batch: [mid in split_ids for mid in batch["mesh_id"]],
            batched=True,
            batch_size=10000,
            num_proc=8,
        )
        print(f"  {split_name}: {len(filtered)} patches, filtered in {time.time()-t1:.1f}s")

        out_dir.mkdir(parents=True, exist_ok=True)
        filtered.save_to_disk(str(out_dir))
        print(f"  Saved to {out_dir}")

    print("Done!")


if __name__ == "__main__":
    main()
