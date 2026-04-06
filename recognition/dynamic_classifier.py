"""
ISL-Multilingual Bridge — Dynamic Sign Classifier
Classifies dynamic (motion-based) ISL signs using Bidirectional LSTM.
"""

import numpy as np
import logging
from pathlib import Path
from typing import Optional, Tuple
from collections import deque

from config.settings import (
    DYNAMIC_MODEL_PATH,
    DYNAMIC_LABEL_ENCODER_PATH,
    FRAME_SEQUENCE_LENGTH,
    SINGLE_HAND_FEATURES,
)

logger = logging.getLogger("isl-bridge.recognition.dynamic")


class DynamicClassifier:
    """
    Classifies dynamic ISL signs from a sequence of frames.
    Uses a Bidirectional LSTM model trained on temporal landmark sequences.

    Input:  (30, 63) — sequence of 30 frames, each with 63 features
    Output: (label: str, confidence: float)
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        label_encoder_path: Optional[Path] = None,
        sequence_length: int = FRAME_SEQUENCE_LENGTH,
    ):
        """
        Args:
            model_path: Path to saved Keras model (.h5)
            label_encoder_path: Path to pickled label encoder
            sequence_length: Number of frames in a sequence (default: 30)
        """
        self.model_path = model_path or DYNAMIC_MODEL_PATH
        self.label_encoder_path = label_encoder_path or DYNAMIC_LABEL_ENCODER_PATH
        self.sequence_length = sequence_length
        self.model = None
        self.label_encoder = None
        self.is_loaded = False

        # Frame buffer for accumulating a sequence
        self.frame_buffer: deque = deque(maxlen=sequence_length)

        self._try_load()

    def _try_load(self):
        """Attempt to load the trained LSTM model and label encoder."""
        try:
            if self.model_path.exists():
                import tensorflow as tf

                self.model = tf.keras.models.load_model(str(self.model_path))
                logger.info(f"Dynamic classifier loaded from {self.model_path}")

                if self.label_encoder_path.exists():
                    import pickle
                    with open(self.label_encoder_path, "rb") as f:
                        self.label_encoder = pickle.load(f)

                self.is_loaded = True
            else:
                logger.warning(
                    f"Dynamic classifier model not found at {self.model_path}. "
                    "Run training/train_dynamic.py first."
                )
        except ImportError:
            logger.warning("TensorFlow not installed. Dynamic classifier unavailable.")
        except Exception as e:
            logger.error(f"Failed to load dynamic classifier: {e}")
            self.is_loaded = False

    def add_frame(self, features: np.ndarray):
        """
        Add a frame's features to the buffer.

        Args:
            features: Normalized landmark feature vector (63,)
        """
        if features is not None and len(features) >= SINGLE_HAND_FEATURES:
            self.frame_buffer.append(features[:SINGLE_HAND_FEATURES])

    def is_buffer_ready(self) -> bool:
        """Check if enough frames have been buffered for prediction."""
        return len(self.frame_buffer) >= self.sequence_length

    def predict(self, sequence: Optional[np.ndarray] = None) -> Tuple[str, float]:
        """
        Predict the sign label from a frame sequence.

        Args:
            sequence: Optional np.ndarray of shape (30, 63).
                      If None, uses the internal frame buffer.

        Returns:
            Tuple of (label: str, confidence: float)
            Returns ("UNKNOWN", 0.0) if model not loaded or buffer not ready
        """
        if not self.is_loaded:
            return ("UNKNOWN", 0.0)

        try:
            if sequence is None:
                if not self.is_buffer_ready():
                    return ("UNKNOWN", 0.0)
                sequence = np.array(list(self.frame_buffer))

            # Ensure correct shape: (1, seq_len, features)
            if sequence.ndim == 2:
                sequence = np.expand_dims(sequence, axis=0)

            # Run inference
            predictions = self.model.predict(sequence, verbose=0)
            predicted_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_idx])

            # Decode label
            if self.label_encoder is not None:
                label = self.label_encoder.inverse_transform([predicted_idx])[0]
            else:
                label = str(predicted_idx)

            return (label, confidence)

        except Exception as e:
            logger.error(f"Dynamic classifier prediction error: {e}")
            return ("UNKNOWN", 0.0)

    def clear_buffer(self):
        """Reset the frame buffer."""
        self.frame_buffer.clear()

    def get_buffer_status(self) -> dict:
        """Get buffer fill status."""
        return {
            "current_frames": len(self.frame_buffer),
            "required_frames": self.sequence_length,
            "is_ready": self.is_buffer_ready(),
            "fill_percent": len(self.frame_buffer) / self.sequence_length * 100,
        }
