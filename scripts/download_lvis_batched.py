"""Download and preprocess LVIS-Wide data in batches to manage disk space.

Strategy:
1. Get LVIS annotations and select categories
2. Split into N batches
3. For each batch: download GLBs → preprocess to patches → clear GLB cache
4. After all batches: save merged metadata
"""
import argparse
import json
import random
import shutil
import os
import gc
from pathlib import Path
from collections import Counter
from tqdm import tqdm

import objaverse

# Import preprocessing functions
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data_prep import load_and_preprocess_mesh
from src.patch_dataset import process_and_save_patches


def select_lvis_wide(lvis, min_per_cat=10, max_per_cat=10, seed=42):
    rng = random.Random(seed)
    selected = {}
    for cat_name, uids in sorted(lvis.items()):
        if len(uids) >= min_per_cat:
            sampled = rng.sample(uids, min(max_per_cat, len(uids)))
            selected[cat_name] = sampled
    return selected


def clear_objaverse_cache():
    cache_dir = Path.home() / ".objaverse"
    if cache_dir.exists():
        size_gb = sum(f.stat().st_size for f in cache_dir.rglob("*") if f.is_file()) / 1e9
        shutil.rmtree(cache_dir)
        print(f"  Cleared cache ({size_gb:.1f} GB)")
    gc.collect()


def get_disk_free_gb():
    stat = os.statvfs("/")
    return (stat.f_bavail * stat.f_frsize) / 1e9


def preprocess_batch(batch_manifest, target_faces, output_root="data",
                     experiment_name="lvis_wide"):
    """Preprocess a batch of downloaded objects into patches."""
    output = Path(output_root)
    metadata = []

    by_cat = {}
    for entry in batch_manifest:
        cat = entry["category"]
        by_cat.setdefault(cat, []).append(entry)

    for cat_name, entries in sorted(by_cat.items()):
        mesh_out = output / "meshes" / experiment_name / cat_name
        patch_out = output / "patches" / experiment_name / cat_name
        mesh_out.mkdir(parents=True, exist_ok=True)

        for entry in tqdm(entries, desc=cat_name, leave=False):
            mesh_id = entry["uid"]
            glb_path = entry["glb_path"]

            # Skip if already processed
            existing = list(Path(str(patch_out)).glob(f"{mesh_id}_patch_*.npz"))
            if existing:
                continue

            try:
                mesh = load_and_preprocess_mesh(glb_path, target_faces=target_faces)
                if mesh is None:
                    continue

                mesh_file = mesh_out / f"{mesh_id}.obj"
                mesh.export(str(mesh_file))

                patch_meta = process_and_save_patches(
                    str(mesh_file), mesh_id, str(patch_out),
                )
                patch_meta["category"] = cat_name
                metadata.append(patch_meta)
            except Exception as e:
                print(f"  Error processing {mesh_id}: {e}")
                continue

    return metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="data/objaverse")
    parser.add_argument("--max_per_cat", type=int, default=10)
    parser.add_argument("--min_per_cat", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_batches", type=int, default=3)
    parser.add_argument("--target_faces", type=int, default=1000)
    args = parser.parse_args()

    out = Path(args.output_dir) / "lvis_wide"
    out.mkdir(parents=True, exist_ok=True)

    print("Loading LVIS annotations...")
    lvis = objaverse.load_lvis_annotations()
    print(f"LVIS: {len(lvis)} categories, {sum(len(v) for v in lvis.values())} objects")

    selected = select_lvis_wide(
        lvis, min_per_cat=args.min_per_cat,
        max_per_cat=args.max_per_cat, seed=args.seed,
    )
    total_objects = sum(len(v) for v in selected.values())
    print(f"Selected: {len(selected)} categories, {total_objects} total objects")

    # Build full item list
    all_items = []
    for cat_name, uids in sorted(selected.items()):
        for uid in uids:
            all_items.append({"uid": uid, "category": cat_name})

    # Split into batches
    batch_size = (len(all_items) + args.n_batches - 1) // args.n_batches
    batches = [all_items[i:i+batch_size] for i in range(0, len(all_items), batch_size)]

    print(f"\nPlan: {len(all_items)} objects in {len(batches)} batches")
    for i, batch in enumerate(batches):
        print(f"  Batch {i+1}: {len(batch)} objects")
    print(f"Disk free: {get_disk_free_gb():.1f} GB\n")

    # Process each batch
    all_manifest = []
    all_metadata = []

    for batch_idx, batch in enumerate(batches):
        print(f"{'='*60}")
        print(f"BATCH {batch_idx+1}/{len(batches)}: {len(batch)} objects")
        print(f"Disk free: {get_disk_free_gb():.1f} GB")
        print(f"{'='*60}")

        if get_disk_free_gb() < 10:
            print("ERROR: Less than 10 GB free, stopping!")
            break

        batch_uids = [item["uid"] for item in batch]
        uid_to_cat = {item["uid"]: item["category"] for item in batch}

        # Download
        print(f"Downloading {len(batch_uids)} objects...")
        objects = objaverse.load_objects(uids=batch_uids)

        batch_manifest = []
        for uid, glb_path in objects.items():
            batch_manifest.append({
                "uid": uid,
                "category": uid_to_cat[uid],
                "glb_path": str(glb_path),
            })
        all_manifest.extend(batch_manifest)
        print(f"Downloaded: {len(batch_manifest)} objects")
        print(f"Disk after download: {get_disk_free_gb():.1f} GB")

        # Preprocess
        print(f"\nPreprocessing batch {batch_idx+1}...")
        batch_metadata = preprocess_batch(
            batch_manifest, target_faces=args.target_faces,
        )
        all_metadata.extend(batch_metadata)
        print(f"Preprocessed: {len(batch_metadata)} meshes → patches")

        # Clear GLB cache
        print("Clearing GLB cache...")
        clear_objaverse_cache()
        print(f"Disk after cleanup: {get_disk_free_gb():.1f} GB")

        # Save progress (metadata so far)
        progress_path = Path("data/patch_metadata_lvis_wide.json")
        with open(progress_path, "w") as f:
            json.dump(all_metadata, f, indent=2)
        print(f"Progress saved: {len(all_metadata)} meshes total\n")

    # Save full manifest
    full_manifest_path = out / "manifest.json"
    with open(full_manifest_path, "w") as f:
        json.dump(all_manifest, f, indent=2)

    cat_counts = Counter(m["category"] for m in all_metadata)
    total_patches = sum(m.get("n_patches", 0) for m in all_metadata)

    print(f"\n{'='*60}")
    print(f"ALL BATCHES COMPLETE")
    print(f"Manifest: {full_manifest_path} ({len(all_manifest)} objects)")
    print(f"Metadata: {len(all_metadata)} meshes, {len(cat_counts)} categories, {total_patches} patches")
    print(f"Disk free: {get_disk_free_gb():.1f} GB")
    print(f"{'='*60}")
    print(f"\nNext step: run category_holdout split:")
    print(f"  PYTHONPATH=. python -c \"")
    print(f"  from scripts.run_preprocessing import split_category_holdout")
    print(f"  import json")
    print(f"  from pathlib import Path")
    print(f"  meta = json.load(open('data/patch_metadata_lvis_wide.json'))")
    print(f"  split_category_holdout(Path('data/patches/lvis_wide'), meta, holdout_categories=50)\"")


if __name__ == "__main__":
    main()
