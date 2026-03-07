"""Batch preprocess ShapeNet meshes and segment into patches."""
import json
import argparse
import random
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
    """Extract model ID from ShapeNet path structure.

    ShapeNet: .../model_id/models/model_normalized.obj → parent.parent.name
    Fallback: .../model_id/model.obj → parent.name
    """
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shapenet_root", type=str, required=True)
    parser.add_argument("--output_root", type=str, default="data")
    parser.add_argument("--target_faces", type=int, default=1000)
    parser.add_argument("--max_per_category", type=int, default=500)
    parser.add_argument("--test_split_ratio", type=float, default=0.2,
                        help="Fraction of train-category meshes held out for testing")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_split", action="store_true",
                        help="Skip train/test split (useful for testing)")
    args = parser.parse_args()

    shapenet = Path(args.shapenet_root)
    output = Path(args.output_root)
    metadata = []

    for cat_name, cat_id in SHAPENET_CATEGORIES.items():
        cat_dir = shapenet / cat_id
        if not cat_dir.exists():
            # Also try direct category name (for non-ShapeNet dirs)
            cat_dir = shapenet
            obj_candidates = sorted(cat_dir.rglob("*.obj"))
            if not obj_candidates:
                print(f"Skipping {cat_name}: {shapenet / cat_id} not found")
                continue
        else:
            obj_candidates = sorted(cat_dir.rglob("*.obj"))

        obj_files = obj_candidates[:args.max_per_category]
        if not obj_files:
            print(f"Skipping {cat_name}: no OBJ files found")
            continue

        mesh_out = output / "meshes" / cat_name
        patch_out = output / "patches" / cat_name
        mesh_out.mkdir(parents=True, exist_ok=True)

        print(f"\n[{cat_name}] Processing {len(obj_files)} meshes...")

        for obj_file in tqdm(obj_files, desc=cat_name):
            mesh_id = extract_mesh_id(obj_file)
            mesh = load_and_preprocess_mesh(str(obj_file), target_faces=args.target_faces)
            if mesh is None:
                continue

            # Save preprocessed mesh
            mesh_file = mesh_out / f"{mesh_id}.obj"
            mesh.export(str(mesh_file))

            # Segment and save patches
            patch_meta = process_and_save_patches(
                str(mesh_file), mesh_id, str(patch_out),
            )
            patch_meta["category"] = cat_name
            metadata.append(patch_meta)

    # Save metadata
    meta_path = output / "patch_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nMetadata saved: {meta_path}")
    print(f"Total meshes processed: {len(metadata)}")

    # Train/test split for training categories
    if not args.no_split:
        patch_dir = output / "patches"
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
