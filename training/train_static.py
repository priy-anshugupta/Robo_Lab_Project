"""
ISL-Multilingual Bridge — Static Classifier Training
Trains ensemble of Random Forest + MLP for static ISL sign classification.

Usage:
    python training/train_static.py
"""

import sys
import pickle
import numpy as np
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import PROCESSED_DATA_DIR, MODELS_DIR

logger = logging.getLogger("isl-bridge.training.static")


def train_static_classifier():
    """Train and save the static sign classifier (Random Forest + MLP ensemble)."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import classification_report, accuracy_score

    processed_dir = Path(PROCESSED_DATA_DIR)
    models_dir = Path(MODELS_DIR)
    models_dir.mkdir(parents=True, exist_ok=True)

    # Load processed data
    try:
        X_train = np.load(processed_dir / "static_X_train.npy", allow_pickle=True)
        y_train = np.load(processed_dir / "static_y_train.npy", allow_pickle=True)
        X_val = np.load(processed_dir / "static_X_val.npy", allow_pickle=True)
        y_val = np.load(processed_dir / "static_y_val.npy", allow_pickle=True)
        X_test = np.load(processed_dir / "static_X_test.npy", allow_pickle=True)
        y_test = np.load(processed_dir / "static_y_test.npy", allow_pickle=True)
    except FileNotFoundError as e:
        logger.error(f"Processed data not found: {e}")
        logger.info("Run 'python training/preprocess.py' first.")
        return

    print(f"\n{'='*60}")
    print(f"  Static Classifier Training")
    print(f"  Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")
    print(f"  Classes: {len(set(y_train))}")
    print(f"{'='*60}\n")

    # Encode labels
    label_encoder = LabelEncoder()
    y_train_enc = label_encoder.fit_transform(y_train)
    y_val_enc = label_encoder.transform(y_val)
    y_test_enc = label_encoder.transform(y_test)

    # ── Train Random Forest ──────────────────────────────────
    print("Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train_enc)
    rf_val_acc = accuracy_score(y_val_enc, rf.predict(X_val))
    print(f"  RF Validation Accuracy: {rf_val_acc:.4f}")

    # ── Train MLP ────────────────────────────────────────────
    print("Training MLP...")
    mlp = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation="relu",
        solver="adam",
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
    )
    mlp.fit(X_train, y_train_enc)
    mlp_val_acc = accuracy_score(y_val_enc, mlp.predict(X_val))
    print(f"  MLP Validation Accuracy: {mlp_val_acc:.4f}")

    # ── Ensemble (average probabilities) ─────────────────────
    print("\nEvaluating Ensemble on test set...")
    rf_probs = rf.predict_proba(X_test)
    mlp_probs = mlp.predict_proba(X_test)
    ensemble_probs = (rf_probs + mlp_probs) / 2
    ensemble_preds = np.argmax(ensemble_probs, axis=1)

    test_acc = accuracy_score(y_test_enc, ensemble_preds)
    print(f"\n  ✅ Ensemble Test Accuracy: {test_acc:.4f}")

    # Classification report
    target_names = label_encoder.classes_.tolist()
    print("\nClassification Report:")
    print(classification_report(y_test_enc, ensemble_preds, target_names=target_names))

    # ── Save models ──────────────────────────────────────────
    # Save the better individual model as the primary (for single-model inference)
    primary_model = rf if rf_val_acc >= mlp_val_acc else mlp
    model_name = "Random Forest" if rf_val_acc >= mlp_val_acc else "MLP"

    with open(models_dir / "static_classifier.pkl", "wb") as f:
        pickle.dump(primary_model, f)
    print(f"\n💾 Saved primary model ({model_name}) → models/static_classifier.pkl")

    # Save ensemble components separately
    with open(models_dir / "static_rf.pkl", "wb") as f:
        pickle.dump(rf, f)
    with open(models_dir / "static_mlp.pkl", "wb") as f:
        pickle.dump(mlp, f)

    # Save label encoder
    with open(models_dir / "static_label_encoder.pkl", "wb") as f:
        pickle.dump(label_encoder, f)
    print(f"💾 Saved label encoder → models/static_label_encoder.pkl")

    print(f"\n{'='*60}")
    print(f"  Training complete!")
    print(f"  RF Accuracy: {rf_val_acc:.4f} | MLP Accuracy: {mlp_val_acc:.4f}")
    print(f"  Ensemble Test Accuracy: {test_acc:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train_static_classifier()
