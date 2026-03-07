"""Download ShapeNet categories from Hugging Face and extract OBJ files."""
import argparse
import zipfile
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CATEGORIES = {
    "chair":    "03001627",
    "table":    "04379243",
    "airplane": "02691156",
    "car":      "02958343",
    "lamp":     "03636649",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", type=str, default="data/ShapeNetCore.v2")
    parser.add_argument("--categories", nargs="+", default=list(CATEGORIES.keys()),
                        help="Categories to download (default: all 5)")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="HF cache directory (default: ~/.cache/huggingface)")
    args = parser.parse_args()

    output = Path(args.output_root)
    output.mkdir(parents=True, exist_ok=True)

    for cat_name in args.categories:
        if cat_name not in CATEGORIES:
            print(f"Unknown category: {cat_name}")
            continue
        cat_id = CATEGORIES[cat_name]
        zip_name = f"{cat_id}.zip"

        print(f"\n{'='*60}")
        print(f"Downloading {cat_name} ({cat_id})...")
        print(f"{'='*60}")

        try:
            zip_path = hf_hub_download(
                repo_id="ShapeNet/ShapeNetCore",
                filename=zip_name,
                repo_type="dataset",
                cache_dir=args.cache_dir,
            )
        except Exception as e:
            print(f"ERROR downloading {cat_name}: {e}")
            print("Make sure you have accepted the dataset terms at:")
            print("  https://huggingface.co/datasets/ShapeNet/ShapeNetCore")
            print("And logged in with: huggingface-cli login")
            continue

        # Extract only model_normalized.obj files
        cat_out = output / cat_id
        print(f"Extracting OBJ files to {cat_out}...")
        n_extracted = 0

        with zipfile.ZipFile(zip_path, 'r') as zf:
            for member in zf.namelist():
                if member.endswith("model_normalized.obj"):
                    zf.extract(member, str(output))
                    n_extracted += 1

        print(f"Extracted {n_extracted} OBJ files for {cat_name}")

    # Summary
    print(f"\n{'='*60}")
    print("Download complete. Summary:")
    for cat_name in args.categories:
        cat_id = CATEGORIES.get(cat_name)
        if cat_id:
            cat_dir = output / cat_id
            n_objs = len(list(cat_dir.rglob("model_normalized.obj"))) if cat_dir.exists() else 0
            print(f"  {cat_name} ({cat_id}): {n_objs} models")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
