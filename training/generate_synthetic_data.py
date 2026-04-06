"""
ISL-Multilingual Bridge — Synthetic Data Generator
Generates synthetic hand landmark data for training when real webcam data
is not available. Creates realistic, class-separable 21-landmark × 3-coord
feature vectors for all ISL sign labels.

Usage:
    python training/generate_synthetic_data.py
"""

import sys
import csv
import json
import numpy as np
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import RAW_DATA_DIR, SINGLE_HAND_FEATURES, FRAME_SEQUENCE_LENGTH

logger = logging.getLogger("isl-bridge.training.synthetic")

# ── Realistic base hand poses (21 landmarks × 3 coords) ────────────
# These are rough normalized positions for various hand shapes.
# Each pose is (21, 3) representing wrist-relative, scale-normalized coords.

def _make_base_poses():
    """
    Create a bank of distinct base hand poses that can be varied with noise
    to generate separable training samples for different ISL sign labels.
    """
    rng = np.random.RandomState(42)

    # Canonical hand skeleton (open palm, all fingers extended)
    open_palm = np.array([
        # Wrist
        [0.0, 0.0, 0.0],
        # Thumb (landmarks 1-4)
        [-0.15, -0.05, -0.03], [-0.25, -0.12, -0.05],
        [-0.32, -0.18, -0.06], [-0.38, -0.22, -0.04],
        # Index finger (5-8)
        [-0.10, -0.28, 0.0], [-0.12, -0.42, 0.0],
        [-0.13, -0.52, 0.0], [-0.13, -0.58, 0.0],
        # Middle finger (9-12)
        [0.0, -0.30, 0.0], [0.0, -0.45, 0.0],
        [0.0, -0.56, 0.0], [0.0, -0.62, 0.0],
        # Ring finger (13-16)
        [0.10, -0.28, 0.0], [0.10, -0.40, 0.0],
        [0.10, -0.50, 0.0], [0.10, -0.55, 0.0],
        # Pinky (17-20)
        [0.20, -0.24, 0.0], [0.22, -0.34, 0.0],
        [0.22, -0.42, 0.0], [0.22, -0.46, 0.0],
    ], dtype=np.float32)

    # Closed fist
    fist = np.array([
        [0.0, 0.0, 0.0],
        [-0.10, -0.03, -0.04], [-0.15, -0.05, -0.08],
        [-0.12, -0.02, -0.12], [-0.08, 0.02, -0.10],
        [-0.08, -0.18, -0.01], [-0.06, -0.15, -0.10],
        [-0.04, -0.10, -0.14], [-0.03, -0.06, -0.12],
        [0.0, -0.20, 0.0], [0.02, -0.16, -0.10],
        [0.03, -0.10, -0.14], [0.03, -0.06, -0.12],
        [0.08, -0.18, 0.0], [0.09, -0.14, -0.10],
        [0.09, -0.09, -0.13], [0.08, -0.05, -0.10],
        [0.16, -0.14, 0.0], [0.17, -0.10, -0.08],
        [0.17, -0.06, -0.11], [0.16, -0.03, -0.08],
    ], dtype=np.float32)

    # Pointing (index extended, rest curled)
    pointing = open_palm.copy()
    for i in [1, 2, 3, 4]:  # curl thumb
        pointing[i] = fist[i]
    for i in [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]:  # curl other fingers
        pointing[i] = fist[i]

    # Victory / Peace (index + middle extended)
    peace = fist.copy()
    for i in [5, 6, 7, 8]:  # extend index
        peace[i] = open_palm[i]
    for i in [9, 10, 11, 12]:  # extend middle
        peace[i] = open_palm[i]

    # Thumbs up
    thumbs_up = fist.copy()
    thumbs_up[1] = [0.0, -0.08, -0.04]
    thumbs_up[2] = [0.0, -0.18, -0.06]
    thumbs_up[3] = [0.0, -0.28, -0.06]
    thumbs_up[4] = [0.0, -0.34, -0.04]

    # "L" shape (thumb + index)
    l_shape = fist.copy()
    for i in [1, 2, 3, 4]:
        l_shape[i] = open_palm[i]
    for i in [5, 6, 7, 8]:
        l_shape[i] = open_palm[i]

    # Pinch (thumb tip touches index tip)
    pinch = open_palm.copy()
    pinch[4] = [-0.13, -0.55, -0.02]  # thumb tip near index tip
    pinch[8] = [-0.13, -0.58, 0.0]

    # "C" shape (curved open hand)
    c_shape = open_palm.copy()
    c_shape[:, 0] *= 0.7  # compress horizontally
    c_shape[4] = [-0.22, -0.30, -0.06]

    # Three fingers (index + middle + ring)
    three = fist.copy()
    for i in [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]:
        three[i] = open_palm[i]

    base_poses = [
        open_palm, fist, pointing, peace, thumbs_up,
        l_shape, pinch, c_shape, three,
    ]

    # Generate more variations by rotating and scaling base poses
    more_poses = []
    for pose in base_poses:
        # Slight rotation around z-axis
        for angle_deg in [15, -15, 30, -30]:
            angle = np.radians(angle_deg)
            rot = np.array([
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle),  np.cos(angle), 0],
                [0, 0, 1],
            ], dtype=np.float32)
            rotated = pose @ rot.T
            more_poses.append(rotated)

        # Mirror
        mirrored = pose.copy()
        mirrored[:, 0] *= -1
        more_poses.append(mirrored)

    all_poses = base_poses + more_poses
    return all_poses


def _normalize_pose(pose: np.ndarray) -> np.ndarray:
    """Normalize a (21,3) pose: translate wrist to origin, scale by bbox diagonal."""
    p = pose.copy()
    p -= p[0]  # wrist to origin
    bbox_diag = np.linalg.norm(p.max(axis=0) - p.min(axis=0))
    if bbox_diag > 1e-6:
        p /= bbox_diag
    return p.flatten().astype(np.float32)


def generate_static_data(
    signs: list,
    samples_per_sign: int = 300,
    output_dir: Path = RAW_DATA_DIR,
    seed: int = 42,
):
    """
    Generate synthetic static sign data and save as CSV files.

    Each sign gets a unique base pose (or combination) with controlled noise
    to ensure classes are separable while still having realistic variation.
    """
    rng = np.random.RandomState(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_base_poses = _make_base_poses()
    n_poses = len(all_base_poses)

    # Assign each sign a unique pose index (wrapping around if needed)
    # Plus a unique per-class offset vector for even more separability
    class_offsets = {}
    for i, sign in enumerate(signs):
        class_offsets[sign] = rng.randn(21, 3).astype(np.float32) * 0.05

    headers = [f"lm_{i}_{axis}" for i in range(21) for axis in ["x", "y", "z"]]
    headers.append("label")

    total_samples = 0

    for sign_idx, sign in enumerate(signs):
        base_pose = all_base_poses[sign_idx % n_poses].copy()
        offset = class_offsets[sign]

        output_file = output_dir / f"{sign}_static.csv"
        rows = []

        for sample in range(samples_per_sign):
            # Add noise + class-specific offset
            noise = rng.randn(21, 3).astype(np.float32) * 0.015
            pose = base_pose + offset + noise

            # Random small rotation for augmentation
            angle = rng.uniform(-10, 10)
            rad = np.radians(angle)
            rot = np.array([
                [np.cos(rad), -np.sin(rad), 0],
                [np.sin(rad),  np.cos(rad), 0],
                [0, 0, 1],
            ], dtype=np.float32)
            pose = pose @ rot.T

            # Random scale variation
            scale = rng.uniform(0.85, 1.15)
            pose *= scale

            features = _normalize_pose(pose)
            row = features.tolist() + [sign]
            rows.append(row)

        # Write CSV
        with open(output_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(rows)

        total_samples += len(rows)
        print(f"  ✅ {sign}: {len(rows)} samples → {output_file.name}")

    print(f"\n💾 Total static samples: {total_samples}")
    return total_samples


def generate_dynamic_data(
    signs: list,
    sequences_per_sign: int = 80,
    sequence_length: int = FRAME_SEQUENCE_LENGTH,
    output_dir: Path = RAW_DATA_DIR,
    seed: int = 123,
):
    """
    Generate synthetic dynamic sign data (multi-frame sequences).
    Each sign gets a unique motion pattern (trajectory through pose space).
    """
    rng = np.random.RandomState(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_base_poses = _make_base_poses()
    n_poses = len(all_base_poses)

    total_sequences = 0

    for sign_idx, sign in enumerate(signs):
        # Pick start and end poses for the motion trajectory
        start_pose = all_base_poses[sign_idx % n_poses].copy()
        end_pose = all_base_poses[(sign_idx + 3) % n_poses].copy()

        # Add unique class offset
        class_offset = rng.randn(21, 3).astype(np.float32) * 0.04

        sequences = []

        for seq in range(sequences_per_sign):
            frames = []
            for t in range(sequence_length):
                # Interpolate between start and end pose
                alpha = t / (sequence_length - 1)
                # Add a sinusoidal motion component for natural movement
                alpha_smooth = 0.5 * (1 - np.cos(np.pi * alpha))
                pose = start_pose * (1 - alpha_smooth) + end_pose * alpha_smooth

                # Add class offset + noise
                noise = rng.randn(21, 3).astype(np.float32) * 0.012
                pose = pose + class_offset + noise

                features = _normalize_pose(pose)
                frames.append(features)

            sequences.append(np.array(frames))  # (30, 63)

        sequences_arr = np.array(sequences)  # (N, 30, 63)
        output_file = output_dir / f"{sign}_dynamic.npy"
        np.save(output_file, sequences_arr)

        total_sequences += len(sequences)
        print(f"  ✅ {sign}: {len(sequences)} sequences → {output_file.name}")

    print(f"\n💾 Total dynamic sequences: {total_sequences}")
    return total_sequences


def main():
    """Generate all synthetic data for all ISL signs from sign list."""
    logging.basicConfig(level=logging.INFO)

    # Load sign list
    sign_list_path = Path(__file__).resolve().parent.parent / "data" / "isl_sign_list.json"
    with open(sign_list_path) as f:
        sign_data = json.load(f)

    # Collect all signs by type
    static_signs = []
    dynamic_signs = []

    for cat_name, cat in sign_data["categories"].items():
        signs = cat["signs"]
        cat_type = cat["type"]

        if cat_type == "static":
            static_signs.extend(signs)
        elif cat_type == "dynamic":
            dynamic_signs.extend(signs)
        elif cat_type == "mixed":
            static_signs.extend(signs)
            dynamic_signs.extend(signs)

    # Remove duplicates while preserving order
    static_signs = list(dict.fromkeys(static_signs))
    dynamic_signs = list(dict.fromkeys(dynamic_signs))

    print(f"\n{'='*60}")
    print(f"  ISL Synthetic Data Generator")
    print(f"  Static signs: {len(static_signs)}")
    print(f"  Dynamic signs: {len(dynamic_signs)}")
    print(f"{'='*60}\n")

    # Generate static data
    print("📦 Generating static sign data...")
    generate_static_data(static_signs, samples_per_sign=300)

    print()

    # Generate dynamic data
    print("📦 Generating dynamic sign data...")
    generate_dynamic_data(dynamic_signs, sequences_per_sign=80)

    print(f"\n{'='*60}")
    print(f"  ✅ Synthetic data generation complete!")
    print(f"  Next steps:")
    print(f"    1. python training/preprocess.py")
    print(f"    2. python training/train_static.py")
    print(f"    3. python training/train_dynamic.py")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
