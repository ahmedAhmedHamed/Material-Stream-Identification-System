"""
Run classical feature extraction v2 over the full dataset and save `.npz` files.

Usage (from repo root):
  python standalone_scripts/run_feature_extractor_v2_dataset.py
"""

from __future__ import annotations

from pathlib import Path

from vector_extraction import feature_extractor_v2 as v2
from vector_extraction.augmentation import available_moderate_augmentations


DATASET_DIR = Path("../dataset")
OUT_TRAIN = Path("../features_train.npz")
OUT_VAL = Path("../features_val.npz")
TEST_SIZE = 0.3
SEED = 42
# Per-class target size is median(train_counts) * AUG_MULT.
# Majority classes are undersampled to the target; minority classes are oversampled via augmentation.
AUG_MULT = 1.0

# Augmentation controls:
# - Set AUG_TECHNIQUE to "random" to use albumentations OneOf pipeline
# - Or set it to one of the legacy single-techniques: "rotate", "flip", "brightness", "contrast", "noise"
AUG_TECHNIQUE = "random"
# Only used when AUG_TECHNIQUE == "random"
AUG_INTENSITY = "moderate"  # "light" | "moderate" | "strong"
# Only used when AUG_TECHNIQUE == "random" and AUG_INTENSITY == "moderate".
# This is the full list of available augmentation keys (random sampling pool).
AVAILABLE_AUGMENTATIONS = list(available_moderate_augmentations())
# Remove items from this list to disable them; selection remains random over the remaining keys.
print('AVAILABLE_AUGMENTATIONS: ', AVAILABLE_AUGMENTATIONS)
ENABLED_AUGMENTATIONS = list(['color_jitter', 'elastic', 'gauss_noise', 'gaussian_blur', 'grid_distortion', 'hflip', 'hsv', 'iso_noise', 'motion_blur'])
# removed: , 'optical_distortion', 'perspective', 'rgb_shift', 'rotate_15', 'vflip' 'brightness_contrast', 'clahe', 'coarse_dropout',

def main() -> int:
    if not DATASET_DIR.exists():
        raise SystemExit(f"Dataset folder not found: {DATASET_DIR}")

    image_paths = v2.get_image_paths(str(DATASET_DIR))
    if not image_paths:
        raise SystemExit(f"No images found under: {DATASET_DIR}")

    v2.build_feature_matrix(
        image_paths=image_paths,
        train_output_path=str(OUT_TRAIN),
        val_output_path=str(OUT_VAL),
        test_size=float(TEST_SIZE),
        random_state=int(SEED),
        augmentation_multiplier=float(AUG_MULT),
        augmentation_technique=str(AUG_TECHNIQUE),
        augmentation_intensity=str(AUG_INTENSITY),
        augmentation_groups=None,
        augmentation_allowlist=tuple(ENABLED_AUGMENTATIONS),
    )
    print(f"Saved: {OUT_TRAIN}")
    print(f"Saved: {OUT_VAL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


