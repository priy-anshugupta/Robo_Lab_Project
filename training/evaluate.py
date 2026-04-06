"""
ISL-Multilingual Bridge — Model Evaluation
Generates confusion matrix, accuracy reports, and per-class metrics.

Usage:
    python training/evaluate.py
"""

import sys
import pickle
import numpy as np
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import PROCESSED_DATA_DIR, MODELS_DIR

logger = logging.getLogger("isl-bridge.training.evaluate")


def evaluate_static():
    """Evaluate the static sign classifier on test data."""
    from sklearn.metrics import (
        classification_report, confusion_matrix, accuracy_score
    )

    processed_dir = Path(PROCESSED_DATA_DIR)
    models_dir = Path(MODELS_DIR)

    try:
        X_test = np.load(processed_dir / "static_X_test.npy", allow_pickle=True)
        y_test = np.load(processed_dir / "static_y_test.npy", allow_pickle=True)

        with open(models_dir / "static_classifier.pkl", "rb") as f:
            model = pickle.load(f)
        with open(models_dir / "label_encoder.pkl", "rb") as f:
            label_encoder = pickle.load(f)
    except FileNotFoundError as e:
        logger.error(f"Files not found: {e}")
        return

    y_test_enc = label_encoder.transform(y_test)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test_enc, y_pred)

    print(f"\n{'='*60}")
    print(f"  Static Classifier Evaluation")
    print(f"  Test Samples: {len(X_test)}")
    print(f"  Overall Accuracy: {accuracy:.4f}")
    print(f"{'='*60}\n")

    target_names = label_encoder.classes_.tolist()
    print("Classification Report:")
    print(classification_report(y_test_enc, y_pred, target_names=target_names))

    # Confusion matrix
    cm = confusion_matrix(y_test_enc, y_pred)
    print("Confusion Matrix:")
    print(cm)

    # Save confusion matrix plot
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.figure(figsize=(max(12, len(target_names) * 0.5), max(10, len(target_names) * 0.4)))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=target_names, yticklabels=target_names)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"Static Classifier Confusion Matrix (Accuracy: {accuracy:.2%})")
        plt.tight_layout()

        plot_path = models_dir / "static_confusion_matrix.png"
        plt.savefig(plot_path, dpi=150)
        print(f"\n📊 Confusion matrix saved → {plot_path}")
        plt.close()
    except ImportError:
        logger.warning("matplotlib/seaborn not installed. Skipping plot.")


def evaluate_dynamic():
    """Evaluate the dynamic sign classifier on test data."""

    processed_dir = Path(PROCESSED_DATA_DIR)
    models_dir = Path(MODELS_DIR)

    try:
        import tensorflow as tf
        from sklearn.metrics import classification_report, accuracy_score

        X_test = np.load(processed_dir / "dynamic_X_test.npy", allow_pickle=True)
        y_test = np.load(processed_dir / "dynamic_y_test.npy", allow_pickle=True)

        model = tf.keras.models.load_model(str(models_dir / "dynamic_classifier.h5"))
        with open(models_dir / "label_encoder.pkl", "rb") as f:
            label_encoder = pickle.load(f)
    except FileNotFoundError as e:
        logger.error(f"Files not found: {e}")
        return
    except ImportError:
        logger.error("TensorFlow not installed.")
        return

    y_test_enc = label_encoder.transform(y_test)
    predictions = model.predict(X_test, verbose=0)
    y_pred = np.argmax(predictions, axis=1)
    accuracy = accuracy_score(y_test_enc, y_pred)

    print(f"\n{'='*60}")
    print(f"  Dynamic Classifier Evaluation")
    print(f"  Test Sequences: {len(X_test)}")
    print(f"  Overall Accuracy: {accuracy:.4f}")
    print(f"{'='*60}\n")

    target_names = label_encoder.classes_.tolist()
    print("Classification Report:")
    print(classification_report(y_test_enc, y_pred, target_names=target_names))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("\n" + "="*60)
    print("  ISL-Multilingual Bridge — Model Evaluation")
    print("="*60)

    evaluate_static()
    evaluate_dynamic()
