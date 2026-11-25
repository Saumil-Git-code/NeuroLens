"""
NeuroLens Dataset Deduplication Checker
=======================================
Detects exact and near-duplicate MRI images using Perceptual Hashing (pHash).
Checks for:
  1. Internal duplicates within the Training dataset.
  2. Cross-set data leakage between Training and Testing datasets.

Images with Hamming distance <= threshold (default: 5) are flagged.
Flagged duplicates are saved to ml/results/duplicates.txt so train.py can exclude them.
"""

import argparse
from pathlib import Path
from typing import Dict, List, Set, Tuple
import numpy as np
from PIL import Image

try:
    import imagehash
except ImportError:
    imagehash = None

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def get_image_paths(folder: Path) -> List[Path]:
    """Recursively collects all valid image paths in a folder."""
    if not folder.exists():
        return []
    paths = []
    for ext in VALID_EXTENSIONS:
        paths.extend(folder.rglob(f"*{ext}"))
        paths.extend(folder.rglob(f"*{ext.upper()}"))
    # Return unique sorted paths
    return sorted(list(set(paths)))


def compute_hashes(image_paths: List[Path]) -> Tuple[List[Path], np.ndarray]:
    """
    Computes perceptual hashes (pHash) for all provided image paths.
    Returns the list of successfully hashed image paths and a boolean array of shape (N, 64).
    """
    valid_paths = []
    hash_list = []

    print(f"Computing perceptual hashes for {len(image_paths)} images...")
    for idx, img_path in enumerate(image_paths):
        try:
            with Image.open(img_path) as img:
                # pHash converts image to 8x8 frequency DCT hash (64 bits)
                ph = imagehash.phash(img)
                # ph.hash is an 8x8 boolean array -> flatten to 64 bools
                hash_list.append(ph.hash.flatten())
                valid_paths.append(img_path)
        except Exception as e:
            print(f"Warning: Could not process {img_path.name}: {e}")

        if (idx + 1) % 1000 == 0 or (idx + 1) == len(image_paths):
            print(f"  Processed {idx + 1}/{len(image_paths)} images")

    if not hash_list:
        return [], np.zeros((0, 64), dtype=bool)

    return valid_paths, np.array(hash_list, dtype=bool)


def find_intra_duplicates(
    paths: List[Path],
    hashes: np.ndarray,
    threshold: int = 5,
    chunk_size: int = 500,
) -> Tuple[List[Tuple[Path, Path, int]], Set[Path]]:
    """
    Finds near-duplicates within a single set of images using chunked vectorization.
    Returns:
        pairs: List of (img1, img2, distance)
        files_to_skip: Set of paths to drop (keeping the first occurrence of each duplicate group)
    """
    n = len(paths)
    pairs = []
    files_to_skip = set()

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = hashes[start:end]  # shape (chunk_size, 64)

        # Compute hamming distance: number of differing bits between chunk and all hashes
        # chunk[:, None, :] shape (chunk_size, 1, 64)
        # hashes[None, :, :] shape (1, n, 64)
        # Result shape: (chunk_size, n)
        dists = np.count_nonzero(chunk[:, None, :] != hashes[None, :, :], axis=2)

        for i_rel in range(end - start):
            i = start + i_rel
            # Only consider j > i to avoid duplicate pairs and self-comparisons
            matching_j = np.where((dists[i_rel] <= threshold) & (np.arange(n) > i))[0]
            for j in matching_j:
                dist = int(dists[i_rel, j])
                pairs.append((paths[i], paths[j], dist))
                # Keep the first image (paths[i]), mark the second (paths[j]) for exclusion
                files_to_skip.add(paths[j].resolve())

    return pairs, files_to_skip


def find_leakage_duplicates(
    train_paths: List[Path],
    train_hashes: np.ndarray,
    test_paths: List[Path],
    test_hashes: np.ndarray,
    threshold: int = 5,
    chunk_size: int = 500,
) -> Tuple[List[Tuple[Path, Path, int]], Set[Path]]:
    """
    Finds cross-dataset near-duplicates between Training and Testing sets (data leakage).
    When an image in Training duplicates an image in Testing, the Training image MUST be excluded.
    Returns:
        pairs: List of (train_path, test_path, distance)
        leakage_train_files: Set of Training paths to exclude
    """
    n_train = len(train_paths)
    pairs = []
    leakage_train_files = set()

    for start in range(0, n_train, chunk_size):
        end = min(start + chunk_size, n_train)
        chunk = train_hashes[start:end]

        # Compare chunk of train images against all test images
        dists = np.count_nonzero(chunk[:, None, :] != test_hashes[None, :, :], axis=2)

        for i_rel in range(end - start):
            train_idx = start + i_rel
            matching_test = np.where(dists[i_rel] <= threshold)[0]
            for test_idx in matching_test:
                dist = int(dists[i_rel, test_idx])
                pairs.append((train_paths[train_idx], test_paths[test_idx], dist))
                # Exclude the training copy to prevent test set data leakage
                leakage_train_files.add(train_paths[train_idx].resolve())

    return pairs, leakage_train_files


def main():
    parser = argparse.ArgumentParser(description="NeuroLens MRI Perceptual Hash Deduplication Tool")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="brain mri dataset",
        help="Path to root dataset folder containing Training and Testing directories.",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=5,
        help="Maximum Hamming distance between hashes to consider images near-duplicates (default: 5).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="ml/results/duplicates.txt",
        help="Path where duplicate exclusion list will be saved.",
    )
    args = parser.parse_args()

    if imagehash is None:
        raise ImportError(
            "The 'imagehash' library is required for deduplication. Install with: pip install imagehash"
        )

    # Determine project root and dataset paths
    project_root = Path(__file__).resolve().parent.parent.parent
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = project_root / data_dir

    train_dir = data_dir / "Training"
    test_dir = data_dir / "Testing"

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = project_root / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("NEUROLENS MRI DATASET DEDUPLICATION CHECK")
    print(f"Dataset root: {data_dir}")
    print(f"Hamming threshold: {args.threshold} (lower = stricter match)")
    print("=" * 70)

    # 1. Collect and hash training images
    train_files = get_image_paths(train_dir)
    print(f"\nFound {len(train_files)} images in Training folder.")
    train_paths, train_hashes = compute_hashes(train_files)

    # 2. Check within Training folder
    print("\nScanning for near-duplicates within Training folder...")
    train_pairs, train_skips = find_intra_duplicates(
        train_paths, train_hashes, threshold=args.threshold
    )
    print(f"  -> Found {len(train_pairs)} intra-training duplicate pairs.")
    print(f"  -> {len(train_skips)} redundant training images flagged for removal.")

    # 3. Check for data leakage against Testing folder
    test_files = get_image_paths(test_dir)
    leakage_pairs = []
    leakage_skips = set()
    if test_files:
        print(f"\nFound {len(test_files)} images in Testing folder.")
        test_paths, test_hashes = compute_hashes(test_files)
        print("\nScanning for cross-set data leakage (Training vs Testing)...")
        leakage_pairs, leakage_skips = find_leakage_duplicates(
            train_paths, train_hashes, test_paths, test_hashes, threshold=args.threshold
        )
        print(f"  -> Found {len(leakage_pairs)} data leakage pairs.")
        print(f"  -> {len(leakage_skips)} training images match test images and must be dropped.")
    else:
        print("\nNo Testing folder found; skipping cross-dataset leakage check.")

    # Combine all training files to skip
    all_files_to_skip = sorted(list(train_skips | leakage_skips), key=lambda p: str(p))

    # Print summary report
    print("\n" + "=" * 70)
    print("DEDUPLICATION SUMMARY REPORT")
    print("=" * 70)
    print(f"Total Training images analyzed:   {len(train_paths)}")
    print(f"Total Testing images analyzed:    {len(test_files)}")
    print(f"Intra-training duplicate pairs:   {len(train_pairs)}")
    print(f"Train/Test leakage pairs:         {len(leakage_pairs)}")
    print(f"Total training images to exclude: {len(all_files_to_skip)}")
    print(f"Writing exclusion list to:        {output_path}")
    print("=" * 70)

    # Write report and exclusion list
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# ======================================================================\n")
        f.write("# NEUROLENS DATASET DEDUPLICATION REPORT\n")
        f.write(f"# Threshold (Hamming distance): <= {args.threshold}\n")
        f.write(f"# Total Training Images: {len(train_paths)}\n")
        f.write(f"# Total Testing Images:  {len(test_files)}\n")
        f.write(f"# Total Duplicate Training Images Flagged: {len(all_files_to_skip)}\n")
        f.write("# Lines below that do NOT start with '#' are paths to exclude from training.\n")
        f.write("# ======================================================================\n\n")

        f.write("# --- DETECTED INTRA-TRAINING DUPLICATE PAIRS ---\n")
        for p1, p2, dist in train_pairs[:100]:  # preview top 100
            f.write(f"# [Dist={dist}] {p1.name} <==> {p2.name}\n")
        if len(train_pairs) > 100:
            f.write(f"# ... and {len(train_pairs) - 100} more pairs\n")

        f.write("\n# --- DETECTED TRAIN/TEST LEAKAGE PAIRS ---\n")
        for p1, p2, dist in leakage_pairs[:100]:
            f.write(f"# [Dist={dist}] Train: {p1.name} <==> Test: {p2.name}\n")
        if len(leakage_pairs) > 100:
            f.write(f"# ... and {len(leakage_pairs) - 100} more pairs\n")

        f.write("\n# --- EXCLUSION LIST (Read by train.py) ---\n")
        for skip_path in all_files_to_skip:
            # Write both relative and absolute path representations for robust matching
            f.write(f"{skip_path}\n")

    print(f"\nDeduplication complete. Exclusions saved to {output_path.name}")


if __name__ == "__main__":
    main()
