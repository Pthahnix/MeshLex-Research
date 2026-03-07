"""Batch preprocess meshes and segment into patches.

Supports two input modes:
1. ShapeNet directory: --shapenet_root (legacy)
2. Manifest JSON: --input_manifest (Objaverse or any source)

Supports two split modes:
1. Default (5cat): train categories get 80/20 mesh-level split
2. category_holdout: randomly hold out N categories as unseen test
"""
import json
import argparse
import random
import shutil
from pathlib import Path
from tqdm import tqdm

from src.data_prep import load_and_preprocess_mesh
from src.patch_dataset import process_and_save_patches

SHAPENET_CATEGORIES = {
    "chair":    "03001627",
    "table":    "04379243",
    "airplane": "02691156",
    "car":      "02958343",
    "lamp":     "03636649",
}

TRAIN_CATEGORIES = {"chair", "table", "airplane"}


def extract_mesh_id(obj_file: Path) -> str:
    """Extract model ID from ShapeNet path structure."""
    if obj_file.parent.name == "models":
        return obj_file.parent.parent.name
    return obj_file.parent.name


def split_patches_by_mesh(patch_dir: Path, category: str, metadata_entries: list,
                          test_ratio: float = 0.2, seed: int = 42):
    """Move patches into _train and _test subdirs based on mesh_id split."""
    rng = random.Random(seed)
    mesh_ids = list(set(
        m["mesh_id"] for m in metadata_entries if m.get("category") == category
    ))
    rng.shuffle(mesh_ids)
    n_test = max(1, int(len(mesh_ids) * test_ratio))
    test_ids = set(mesh_ids[:n_test])

    patch_path = patch_dir / category
    train_path = patch_dir / f"{category}_train"
    test_path = patch_dir / f"{category}_test"
    train_path.mkdir(parents=True, exist_ok=True)
    test_path.mkdir(parents=True, exist_ok=True)

    n_train, n_test_patches = 0, 0
    for npz_file in sorted(patch_path.glob("*.npz")):
        file_mesh_id = npz_file.stem.rsplit("_patch_", 1)[0]
        if file_mesh_id in test_ids:
            npz_file.rename(test_path / npz_file.name)
            n_test_patches += 1
        else:
            npz_file.rename(train_path / npz_file.name)
            n_train += 1

    # Remove now-empty original dir
    if patch_path.exists() and not any(patch_path.glob("*.npz")):
        patch_path.rmdir()

    n_train_meshes = len(mesh_ids) - len(test_ids)
    n_test_meshes = len(test_ids)
    print(f"  {category}: {n_train_meshes} train meshes ({n_train} patches), "
          f"{n_test_meshes} test meshes ({n_test_patches} patches)")
    return n_train_meshes, n_test_meshes


def split_category_holdout(patch_dir: Path, metadata: list,
                           holdout_categories: int = 50,
                           test_ratio: float = 0.2, seed: int = 42):
    """Split by holding out entire categories as unseen, plus mesh-level split for seen."""
    rng = random.Random(seed)

    all_cats = sorted(set(m["category"] for m in metadata))
    rng.shuffle(all_cats)

    n_holdout = min(holdout_categories, len(all_cats) // 5)
    unseen_cats = set(all_cats[:n_holdout])
    seen_cats = set(all_cats[n_holdout:])

    print(f"\nCategory holdout split:")
    print(f"  Seen categories: {len(seen_cats)}")
    print(f"  Unseen categories: {len(unseen_cats)}")

    seen_train = patch_dir / "seen_train"
    seen_test = patch_dir / "seen_test"
    unseen_dir = patch_dir / "unseen"
    seen_train.mkdir(parents=True, exist_ok=True)
    seen_test.mkdir(parents=True, exist_ok=True)
    unseen_dir.mkdir(parents=True, exist_ok=True)

    # For seen categories: mesh-level 80/20 split
    seen_meta = [m for m in metadata if m["category"] in seen_cats]
    seen_mesh_ids = list(set(m["mesh_id"] for m in seen_meta))
    rng.shuffle(seen_mesh_ids)
    n_seen_test = max(1, int(len(seen_mesh_ids) * test_ratio))
    seen_test_ids = set(seen_mesh_ids[:n_seen_test])

    n_train_patches, n_test_patches, n_unseen_patches = 0, 0, 0

    for cat in all_cats:
        cat_dir = patch_dir / cat
        if not cat_dir.exists():
            continue
        for npz_file in sorted(cat_dir.glob("*.npz")):
            file_mesh_id = npz_file.stem.rsplit("_patch_", 1)[0]
            if cat in unseen_cats:
                shutil.move(str(npz_file), str(unseen_dir / npz_file.name))
                n_unseen_patches += 1
            elif file_mesh_id in seen_test_ids:
                shutil.move(str(npz_file), str(seen_test / npz_file.name))
                n_test_patches += 1
            else:
                shutil.move(str(npz_file), str(seen_train / npz_file.name))
                n_train_patches += 1
        # Clean up empty category dir
        if cat_dir.exists() and not any(cat_dir.glob("*.npz")):
            cat_dir.rmdir()

    print(f"  seen_train: {n_train_patches} patches")
    print(f"  seen_test: {n_test_patches} patches")
    print(f"  unseen: {n_unseen_patches} patches ({len(unseen_cats)} categories)")


def process_from_manifest(manifest_path: str, experiment_name: str,
                          output_root: str, target_faces: int,
                          max_per_category: int = 0):
    """Process meshes listed in a manifest JSON file."""
    with open(manifest_path) as f:
        manifest = json.load(f)

    output = Path(output_root)
    metadata = []

    # Group by category
    by_cat = {}
    for entry in manifest:
        cat = entry["category"]
        by_cat.setdefault(cat, []).append(entry)

    for cat_name, entries in sorted(by_cat.items()):
        if max_per_category > 0:
            entries = entries[:max_per_category]

        mesh_out = output / "meshes" / experiment_name / cat_name
        patch_out = output / "patches" / experiment_name / cat_name
        mesh_out.mkdir(parents=True, exist_ok=True)

        print(f"\n[{cat_name}] Processing {len(entries)} meshes...")

        for entry in tqdm(entries, desc=cat_name):
            mesh_id = entry["uid"]
            glb_path = entry["glb_path"]

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

    return metadata


def process_from_shapenet(shapenet_root: str, output_root: str,
                          target_faces: int, max_per_category: int):
    """Process meshes from ShapeNet directory structure (legacy)."""
    shapenet = Path(shapenet_root)
    output = Path(output_root)
    metadata = []

    for cat_name, cat_id in SHAPENET_CATEGORIES.items():
        cat_dir = shapenet / cat_id
        if not cat_dir.exists():
            cat_dir = shapenet
            obj_candidates = sorted(cat_dir.rglob("*.obj"))
            if not obj_candidates:
                print(f"Skipping {cat_name}: {shapenet / cat_id} not found")
                continue
        else:
            obj_candidates = sorted(cat_dir.rglob("*.obj"))

        obj_files = obj_candidates[:max_per_category]
        if not obj_files:
            print(f"Skipping {cat_name}: no OBJ files found")
            continue

        mesh_out = output / "meshes" / cat_name
        patch_out = output / "patches" / cat_name
        mesh_out.mkdir(parents=True, exist_ok=True)

        print(f"\n[{cat_name}] Processing {len(obj_files)} meshes...")

        for obj_file in tqdm(obj_files, desc=cat_name):
            mesh_id = extract_mesh_id(obj_file)
            mesh = load_and_preprocess_mesh(str(obj_file), target_faces=target_faces)
            if mesh is None:
                continue

            mesh_file = mesh_out / f"{mesh_id}.obj"
            mesh.export(str(mesh_file))

            patch_meta = process_and_save_patches(
                str(mesh_file), mesh_id, str(patch_out),
            )
            patch_meta["category"] = cat_name
            metadata.append(patch_meta)

    return metadata


def main():
    parser = argparse.ArgumentParser()
    # Input mode (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input_manifest", type=str,
                             help="Manifest JSON from download_objaverse.py")
    input_group.add_argument("--shapenet_root", type=str,
                             help="ShapeNet directory root (legacy)")

    parser.add_argument("--experiment_name", type=str, default="",
                        help="Experiment name for output subdirectory (e.g., 5cat, lvis_wide)")
    parser.add_argument("--output_root", type=str, default="data")
    parser.add_argument("--target_faces", type=int, default=1000)
    parser.add_argument("--max_per_category", type=int, default=0,
                        help="Max meshes per category (0 = no limit)")
    parser.add_argument("--test_split_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_split", action="store_true")

    # Split mode
    parser.add_argument("--split_mode", choices=["default", "category_holdout"],
                        default="default")
    parser.add_argument("--holdout_categories", type=int, default=50,
                        help="Number of categories to hold out (category_holdout mode)")
    args = parser.parse_args()

    output = Path(args.output_root)

    # Process meshes
    if args.input_manifest:
        metadata = process_from_manifest(
            args.input_manifest, args.experiment_name,
            args.output_root, args.target_faces, args.max_per_category,
        )
    else:
        metadata = process_from_shapenet(
            args.shapenet_root, args.output_root,
            args.target_faces, args.max_per_category,
        )

    # Save metadata
    suffix = f"_{args.experiment_name}" if args.experiment_name else ""
    meta_path = output / f"patch_metadata{suffix}.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nMetadata saved: {meta_path}")
    print(f"Total meshes processed: {len(metadata)}")

    # Split
    if not args.no_split and metadata:
        if args.experiment_name:
            patch_dir = output / "patches" / args.experiment_name
        else:
            patch_dir = output / "patches"

        if args.split_mode == "category_holdout":
            split_category_holdout(
                patch_dir, metadata,
                holdout_categories=args.holdout_categories,
                test_ratio=args.test_split_ratio, seed=args.seed,
            )
        else:
            # Default: split train categories by mesh_id
            print(f"\nSplitting train categories (ratio={args.test_split_ratio}):")
            for cat_name in TRAIN_CATEGORIES:
                cat_entries = [m for m in metadata if m.get("category") == cat_name]
                if cat_entries and (patch_dir / cat_name).exists():
                    split_patches_by_mesh(
                        patch_dir, cat_name, metadata,
                        test_ratio=args.test_split_ratio, seed=args.seed,
                    )


if __name__ == "__main__":
    main()
