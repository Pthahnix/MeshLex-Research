"""Validate HF dataset meets spec thresholds via Daft.

Usage:
    python scripts/validate_dataset_daft.py --hf_repo Pthahnix/MeshLex-Patches
"""
import argparse
import json
import logging
from collections import Counter
from pathlib import Path

import daft
from huggingface_hub import hf_hub_download

from src.daft_utils import get_hf_io_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_repo", default="Pthahnix/MeshLex-Patches")
    parser.add_argument("--work_dir", default="/tmp/meshlex_validate")
    args = parser.parse_args()

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    io_config = get_hf_io_config()

    # Download metadata + splits
    for fname in ["metadata.json", "splits.json", "stats.json"]:
        hf_hub_download(repo_id=args.hf_repo, filename=fname,
                        repo_type="dataset", local_dir=str(work_dir))
    with open(work_dir / "metadata.json") as f:
        metadata = json.load(f)
    with open(work_dir / "splits.json") as f:
        splits = json.load(f)

    checks = []

    # Metadata-based checks
    n_obj = sum(1 for m in metadata.values() if m["source"] == "objaverse")
    n_sn = sum(1 for m in metadata.values() if m["source"] == "shapenet")
    n_total = len(metadata)
    n_patches = sum(m["n_patches"] for m in metadata.values())
    n_cats = len(set(m["category"] for m in metadata.values()))

    checks.append(("Objaverse meshes >= 35,000", n_obj, n_obj >= 35000))
    checks.append(("ShapeNet meshes >= 45,000", n_sn, n_sn >= 45000))
    checks.append(("Total meshes >= 75,000", n_total, n_total >= 75000))
    checks.append(("Total patches >= 2,500,000", n_patches, n_patches >= 2_500_000))
    checks.append(("Categories >= 500", n_cats, n_cats >= 500))

    all_split_ids = set(splits["seen_train"] + splits["seen_test"] + splits["unseen"])
    checks.append(("All meshes in splits", len(all_split_ids), all_split_ids == set(metadata.keys())))

    # Daft-based Parquet validation
    log.info("Reading Parquet from HF via Daft...")
    try:
        df = daft.read_parquet(
            f"hf://datasets/{args.hf_repo}/**/*.parquet", io_config=io_config,
        )
        parquet_rows = df.count_rows()
        checks.append(("Parquet rows == metadata patches", parquet_rows,
                        parquet_rows == n_patches))

        # Sample 10 rows and check columns
        sample = df.limit(10).collect()
        schema = df.schema()
        expected_cols = [
            "mesh_id", "patch_idx", "category", "source",
            "n_faces", "n_verts", "faces", "vertices",
            "local_vertices", "local_vertices_nopca",
            "centroid", "principal_axes", "scale",
            "boundary_vertices", "global_face_indices",
        ]
        missing = [c for c in expected_cols if c not in schema.column_names()]
        checks.append(("All columns present", f"missing={missing}", len(missing) == 0))
        checks.append(("Sample rows fetched", sample.count_rows(), sample.count_rows() == 10))
    except Exception as e:
        checks.append(("Daft Parquet read", str(e), False))

    # Print report
    print("\n" + "=" * 60)
    print("DATASET VALIDATION REPORT")
    print("=" * 60)
    all_pass = True
    for name, value, passed in checks:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  [{status}] {name}: {value}")
    print("=" * 60)
    print("RESULT:", "ALL CHECKS PASSED" if all_pass else "SOME CHECKS FAILED")
    print("=" * 60)


if __name__ == "__main__":
    main()
