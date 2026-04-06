"""
ISL-Multilingual Bridge — Dynamic Classifier Training
Trains Bidirectional LSTM for dynamic ISL sign classification.

Usage:
    python training/train_dynamic.py
"""

import sys
import pickle
import numpy as np
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import PROCESSED_DATA_DIR, MODELS_DIR, FRAME_SEQUENCE_LENGTH, SINGLE_HAND_FEATURES

logger = logging.getLogger("isl-bridge.training.dynamic")


def train_dynamic_classifier():
    """Train and save the LSTM-based dynamic sign classifier."""
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import (
            Bidirectional, LSTM, Dense, Dropout, BatchNormalization
        )
        from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
        from tensorflow.keras.utils import to_categorical
        from sklearn.preprocessing import LabelEncoder
    except ImportError:
        logger.error("TensorFlow not installed. Run: pip install tensorflow")
        return

    processed_dir = Path(PROCESSED_DATA_DIR)
    models_dir = Path(MODELS_DIR)
    models_dir.mkdir(parents=True, exist_ok=True)

    # Load processed data
    try:
        X_train = np.load(processed_dir / "dynamic_X_train.npy", allow_pickle=True)
        y_train = np.load(processed_dir / "dynamic_y_train.npy", allow_pickle=True)
        X_val = np.load(processed_dir / "dynamic_X_val.npy", allow_pickle=True)
        y_val = np.load(processed_dir / "dynamic_y_val.npy", allow_pickle=True)
        X_test = np.load(processed_dir / "dynamic_X_test.npy", allow_pickle=True)
        y_test = np.load(processed_dir / "dynamic_y_test.npy", allow_pickle=True)
    except FileNotFoundError as e:
        logger.error(f"Processed data not found: {e}")
        logger.info("Run 'python training/preprocess.py' first.")
        return

    print(f"\n{'='*60}")
    print(f"  Dynamic Classifier Training (Bidirectional LSTM)")
    print(f"  Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")
    print(f"  Sequence shape: ({FRAME_SEQUENCE_LENGTH}, {SINGLE_HAND_FEATURES})")
    print(f"  Classes: {len(set(y_train))}")
    print(f"{'='*60}\n")

    # Encode labels
    label_encoder = LabelEncoder()
    y_train_enc = label_encoder.fit_transform(y_train)
    y_val_enc = label_encoder.transform(y_val)
    y_test_enc = label_encoder.transform(y_test)

    num_classes = len(label_encoder.classes_)
    y_train_cat = to_categorical(y_train_enc, num_classes)
    y_val_cat = to_categorical(y_val_enc, num_classes)
    y_test_cat = to_categorical(y_test_enc, num_classes)

    # ── Build LSTM Model ─────────────────────────────────────
    model = Sequential([
        Bidirectional(
            LSTM(128, return_sequences=True),
            input_shape=(X_train.shape[1], X_train.shape[2])
        ),
        Dropout(0.3),
        BatchNormalization(),

        Bidirectional(LSTM(64, return_sequences=False)),
        Dropout(0.2),
        BatchNormalization(),

        Dense(64, activation="relu"),
        Dropout(0.2),

        Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()

    # ── Callbacks ────────────────────────────────────────────
    callbacks = [
        EarlyStopping(
            monitor="val_accuracy",
            patience=15,
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            str(models_dir / "dynamic_classifier_best.h5"),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
    ]

    # ── Train ────────────────────────────────────────────────
    print("\nTraining LSTM...")
    history = model.fit(
        X_train, y_train_cat,
        validation_data=(X_val, y_val_cat),
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        verbose=1,
    )

    # ── Evaluate ─────────────────────────────────────────────
    test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
    print(f"\n  ✅ Test Accuracy: {test_acc:.4f}")
    print(f"  Test Loss: {test_loss:.4f}")

    # ── Save ─────────────────────────────────────────────────
    model.save(str(models_dir / "dynamic_classifier.h5"))
    print(f"\n💾 Saved model → models/dynamic_classifier.h5")

    # Update shared label encoder (if dynamic has different classes)
    with open(models_dir / "dynamic_label_encoder.pkl", "wb") as f:
        pickle.dump(label_encoder, f)
    print(f"💾 Saved label encoder → models/dynamic_label_encoder.pkl")

    # Save training history
    np.save(models_dir / "dynamic_history.npy", history.history)

    print(f"\n{'='*60}")
    print(f"  Training complete!")
    print(f"  Best Val Accuracy: {max(history.history['val_accuracy']):.4f}")
    print(f"  Test Accuracy: {test_acc:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train_dynamic_classifier()
