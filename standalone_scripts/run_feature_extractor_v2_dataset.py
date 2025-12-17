"""
Run classical feature extraction v2 over the full dataset and save `.npz` files.

Usage (from repo root):
  python standalone_scripts/run_feature_extractor_v2_dataset.py
  python standalone_scripts/run_feature_extractor_v2_dataset.py --dataset dataset --out-train features_train_v2.npz --out-val features_val_v2.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

from vector_extraction import feature_extractor_v2 as v2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="dataset", help="Dataset root folder")
    p.add_argument("--out-train", type=str, default="features_train_v2.npz", help="Output train .npz path")
    p.add_argument("--out-val", type=str, default="features_val_v2.npz", help="Output val .npz path")
    p.add_argument("--test-size", type=float, default=0.3, help="Validation split fraction")
    p.add_argument("--seed", type=int, default=42, help="Random seed for train/val split")
    p.add_argument("--aug-mult", type=float, default=1.0, help="Uniform augmentation multiplier for training set")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    dataset_dir = Path(args.dataset)
    if not dataset_dir.exists():
        raise SystemExit(f"Dataset folder not found: {dataset_dir}")

    image_paths = v2.get_image_paths(str(dataset_dir))
    if not image_paths:
        raise SystemExit(f"No images found under: {dataset_dir}")

    v2.build_feature_matrix(
        image_paths=image_paths,
        train_output_path=str(Path(args.out_train)),
        val_output_path=str(Path(args.out_val)),
        test_size=float(args.test_size),
        random_state=int(args.seed),
        augmentation_multiplier=float(args.aug_mult),
    )
    print(f"Saved: {args.out_train}")
    print(f"Saved: {args.out_val}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


