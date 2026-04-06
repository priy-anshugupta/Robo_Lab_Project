"""
ISL-Multilingual Bridge — Static Sign Classifier
Classifies static (non-moving) ISL signs using Random Forest + MLP ensemble.
"""

import numpy as np
import pickle
import logging
from pathlib import Path
from typing import Optional, Tuple

from config.settings import (
    STATIC_MODEL_PATH,
    STATIC_LABEL_ENCODER_PATH,
    SINGLE_HAND_FEATURES,
)

logger = logging.getLogger("isl-bridge.recognition.static")


class StaticClassifier:
    """
    Classifies static ISL signs from a single frame's landmark features.
    Uses an ensemble of Random Forest and MLP classifiers.

    Input:  (63,) normalized landmark vector (single hand)
    Output: (label: str, confidence: float)
    """

    def __init__(self, model_path: Optional[Path] = None, label_encoder_path: Optional[Path] = None):
        """
        Args:
            model_path: Path to pickled model file
            label_encoder_path: Path to pickled label encoder
        """
        self.model_path = model_path or STATIC_MODEL_PATH
        self.label_encoder_path = label_encoder_path or STATIC_LABEL_ENCODER_PATH
        self.model = None
        self.label_encoder = None
        self.is_loaded = False

        self._try_load()

    def _try_load(self):
        """Attempt to load the trained model and label encoder."""
        try:
            if self.model_path.exists() and self.label_encoder_path.exists():
                with open(self.model_path, "rb") as f:
                    self.model = pickle.load(f)
                with open(self.label_encoder_path, "rb") as f:
                    self.label_encoder = pickle.load(f)
                self.is_loaded = True
                logger.info(f"Static classifier loaded from {self.model_path}")
            else:
                logger.warning(
                    f"Static classifier model not found at {self.model_path}. "
                    "Run training/train_static.py first."
                )
        except Exception as e:
            logger.error(f"Failed to load static classifier: {e}")
            self.is_loaded = False

    def _heuristic_predict(self, features: np.ndarray) -> Tuple[str, float]:
        """
        A fallback rule-based classifier if the trained model is missing.
        Uses normalized y-coordinates of finger tips vs lower joints to determine 
        which fingers are extended, then maps shapes to basic ISL signs.
        Note: The features are 63 floats (21 landmarks * 3 coords). 
        Wrist is at origin. y is negative pointing upwards.
        """
        # Reshape to (21, 3)
        landmarks = features.reshape(21, 3)
        
        # Helper to check if a finger is extended.
        # tip is index, lower joint is index-2
        def is_extended(tip_idx):
            # y coordinate: negative is "up" relative to wrist
            return landmarks[tip_idx, 1] < landmarks[tip_idx - 2, 1]
            
        thumb_ext = is_extended(4)
        index_ext = is_extended(8)
        middle_ext = is_extended(12)
        ring_ext = is_extended(16)
        pinky_ext = is_extended(20)
        
        # Count extended fingers (excluding thumb)
        ext_count = sum([index_ext, middle_ext, ring_ext, pinky_ext])
        
        label = "UNKNOWN"
        confidence = 0.5
        
        if ext_count == 4 and thumb_ext:
            label = "HELLO"
            confidence = 0.8
        elif ext_count == 0 and thumb_ext:
            label = "GOOD" # Thumbs up
            confidence = 0.8
        elif ext_count == 0 and not thumb_ext:
            label = "STOP" # Fist
            confidence = 0.8
        elif ext_count == 1 and index_ext and not thumb_ext:
            label = "YOU" # Pointing
            confidence = 0.8
        elif ext_count == 2 and index_ext and middle_ext and not thumb_ext:
            label = "TWO" # Peace sign
            confidence = 0.8
        elif ext_count == 1 and index_ext and thumb_ext:
            label = "L" # L shape
            confidence = 0.8
            
        return label, confidence

    def predict(self, features: np.ndarray) -> Tuple[str, float]:
        """
        Predict the sign label for a single frame.

        Args:
            features: Normalized landmark feature vector of shape (63,)

        Returns:
            Tuple of (label: str, confidence: float)
            Returns ("UNKNOWN", 0.0) if model is not loaded
        """
        if not self.is_loaded:
            return self._heuristic_predict(features)

        try:
            # Ensure correct shape
            if features.shape != (SINGLE_HAND_FEATURES,):
                features = features[:SINGLE_HAND_FEATURES]

            # Reshape for sklearn: (1, 63)
            X = features.reshape(1, -1)

            # Get prediction probabilities
            if hasattr(self.model, "predict_proba"):
                probs = self.model.predict_proba(X)[0]
                predicted_idx = np.argmax(probs)
                confidence = float(probs[predicted_idx])
            else:
                predicted_idx = self.model.predict(X)[0]
                confidence = 1.0  # No probability available

            # Decode label
            if self.label_encoder is not None:
                label = self.label_encoder.inverse_transform([predicted_idx])[0]
            else:
                label = str(predicted_idx)

            return (label, confidence)

        except Exception as e:
            logger.error(f"Static classifier prediction error: {e}")
            return ("UNKNOWN", 0.0)

    def predict_top_k(self, features: np.ndarray, k: int = 5) -> list:
        """
        Get top-k predictions with probabilities.

        Args:
            features: Normalized landmark feature vector (63,)
            k: Number of top predictions to return

        Returns:
            List of (label, confidence) tuples, sorted by confidence descending
        """
        if not self.is_loaded or not hasattr(self.model, "predict_proba"):
            label, conf = self._heuristic_predict(features)
            return [(label, conf)]

        try:
            X = features.reshape(1, -1)
            probs = self.model.predict_proba(X)[0]
            top_indices = np.argsort(probs)[::-1][:k]

            results = []
            for idx in top_indices:
                label = self.label_encoder.inverse_transform([idx])[0]
                results.append((label, float(probs[idx])))

            return results

        except Exception as e:
            logger.error(f"Static classifier top-k error: {e}")
            return [("UNKNOWN", 0.0)]

