"""
ISL-Multilingual Bridge — Data Preprocessing
Cleans, augments, and splits collected landmark data for training.

Usage:
    python training/preprocess.py
"""

import sys
import csv
import numpy as np
import logging
from pathlib import Path
from typing import Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import RAW_DATA_DIR, PROCESSED_DATA_DIR, SINGLE_HAND_FEATURES

logger = logging.getLogger("isl-bridge.training.preprocess")


def load_static_data(raw_dir: Path = RAW_DATA_DIR) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load all static sign CSV files from raw data directory.

    Returns:
        Tuple of (X: np.ndarray, y: np.ndarray) — features and labels
    """
    csv_files = list(raw_dir.glob("*_static.csv"))
    if not csv_files:
        logger.warning(f"No static CSV files found in {raw_dir}")
        return np.array([]), np.array([])

    all_features = []
    all_labels = []

    for csv_file in csv_files:
        with open(csv_file, "r") as f:
            reader = csv.reader(f)
            header = next(reader)  # Skip header

            for row in reader:
                features = [float(x) for x in row[:-1]]
                label = row[-1]
                all_features.append(features)
                all_labels.append(label)

    X = np.array(all_features, dtype=np.float32)
    y = np.array(all_labels)

    logger.info(f"Loaded static data: X={X.shape}, y={y.shape}, classes={len(set(y))}")
    return X, y


def load_dynamic_data(raw_dir: Path = RAW_DATA_DIR) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load all dynamic sign NPY files from raw data directory.

    Returns:
        Tuple of (X: np.ndarray, y: np.ndarray) — sequences and labels
    """
    npy_files = list(raw_dir.glob("*_dynamic.npy"))
    if not npy_files:
        logger.warning(f"No dynamic NPY files found in {raw_dir}")
        return np.array([]), np.array([])

    all_sequences = []
    all_labels = []

    for npy_file in npy_files:
        label = npy_file.stem.replace("_dynamic", "")
        sequences = np.load(npy_file)  # (N, 30, 63)

        for seq in sequences:
            all_sequences.append(seq)
            all_labels.append(label)

    X = np.array(all_sequences, dtype=np.float32)
    y = np.array(all_labels)

    logger.info(f"Loaded dynamic data: X={X.shape}, y={y.shape}, classes={len(set(y))}")
    return X, y


def augment_static(X: np.ndarray, y: np.ndarray, augment_factor: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Augment static data with noise, scaling, and mirroring.

    Args:
        X: Feature array (N, 63)
        y: Label array (N,)
        augment_factor: How many augmented copies per original sample

    Returns:
        Augmented (X, y) with original data included
    """
    augmented_X = [X]
    augmented_y = [y]

    for _ in range(augment_factor):
        # Random noise injection
        noise = np.random.normal(0, 0.01, X.shape).astype(np.float32)
        augmented_X.append(X + noise)
        augmented_y.append(y)

        # Random scaling
        scale = np.random.uniform(0.9, 1.1, (X.shape[0], 1)).astype(np.float32)
        augmented_X.append(X * scale)
        augmented_y.append(y)

    X_aug = np.concatenate(augmented_X, axis=0)
    y_aug = np.concatenate(augmented_y, axis=0)

    logger.info(f"Augmented: {X.shape[0]} → {X_aug.shape[0]} samples")
    return X_aug, y_aug


def augment_dynamic(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Augment dynamic data with time warping and noise.

    Args:
        X: Sequence array (N, 30, 63)
        y: Label array (N,)

    Returns:
        Augmented (X, y)
    """
    augmented_X = [X]
    augmented_y = [y]

    # Random noise
    noise = np.random.normal(0, 0.01, X.shape).astype(np.float32)
    augmented_X.append(X + noise)
    augmented_y.append(y)

    # Time reversal (play sequence backwards)
    augmented_X.append(X[:, ::-1, :])
    augmented_y.append(y)

    X_aug = np.concatenate(augmented_X, axis=0)
    y_aug = np.concatenate(augmented_y, axis=0)

    logger.info(f"Dynamic augmented: {X.shape[0]} → {X_aug.shape[0]} sequences")
    return X_aug, y_aug


def split_data(
    X: np.ndarray, y: np.ndarray, test_ratio: float = 0.2, val_ratio: float = 0.1
) -> dict:
    """
    Split data into train/validation/test sets.

    Args:
        X: Features
        y: Labels
        test_ratio: Fraction for test set
        val_ratio: Fraction for validation set

    Returns:
        dict with 'X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test'
    """
    from sklearn.model_selection import train_test_split

    # First split: train+val vs test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_ratio, random_state=42, stratify=y
    )

    # Second split: train vs val
    val_size = val_ratio / (1 - test_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_size, random_state=42, stratify=y_trainval
    )

    logger.info(
        f"Split: train={X_train.shape[0]}, val={X_val.shape[0]}, test={X_test.shape[0]}"
    )

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
    }


def preprocess_and_save():
    """Full preprocessing pipeline: load → augment → split → save."""
    output_dir = Path(PROCESSED_DATA_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process static data
    X_static, y_static = load_static_data()
    if len(X_static) > 0:
        X_aug, y_aug = augment_static(X_static, y_static)
        splits = split_data(X_aug, y_aug)
        for key, arr in splits.items():
            np.save(output_dir / f"static_{key}.npy", arr)
        logger.info(f"Static data saved to {output_dir}")

    # Process dynamic data
    X_dynamic, y_dynamic = load_dynamic_data()
    if len(X_dynamic) > 0:
        X_aug_d, y_aug_d = augment_dynamic(X_dynamic, y_dynamic)
        splits_d = split_data(X_aug_d, y_aug_d)
        for key, arr in splits_d.items():
            np.save(output_dir / f"dynamic_{key}.npy", arr)
        logger.info(f"Dynamic data saved to {output_dir}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    preprocess_and_save()
