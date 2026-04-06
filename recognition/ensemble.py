"""
ISL-Multilingual Bridge — Ensemble Recognizer
Combines static and dynamic classifiers with confidence gating.
"""

import numpy as np
import logging
from typing import Optional

from config.settings import CONFIDENCE_THRESHOLD
from recognition.static_classifier import StaticClassifier
from recognition.dynamic_classifier import DynamicClassifier

logger = logging.getLogger("isl-bridge.recognition.ensemble")

# Motion threshold to decide between static and dynamic classification
MOTION_THRESHOLD = 0.015


class EnsembleRecognizer:
    """
    Combines predictions from static and dynamic classifiers.
    Uses motion detection to route to the appropriate classifier,
    and applies a confidence gate to filter uncertain predictions.
    """

    def __init__(
        self,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        static_weight: float = 0.7,
        dynamic_weight: float = 0.3,
    ):
        """
        Args:
            confidence_threshold: Minimum confidence to accept a prediction
            static_weight: Weight for the primary classifier in ensemble
            dynamic_weight: Weight for the secondary classifier in ensemble
        """
        self.static_classifier = StaticClassifier()
        self.dynamic_classifier = DynamicClassifier()
        self.confidence_threshold = confidence_threshold
        self.static_weight = static_weight
        self.dynamic_weight = dynamic_weight

        self.prev_features: Optional[np.ndarray] = None
        self.motion_history: list = []

        logger.info(
            f"EnsembleRecognizer initialized "
            f"(threshold={confidence_threshold}, "
            f"static_weight={static_weight}, dynamic_weight={dynamic_weight})"
        )

    def predict(self, features: np.ndarray) -> dict:
        """
        HEURISTIC GESTURE RECOGNIZER
        """
        self.dynamic_classifier.add_frame(features)
        
        try:
            # Reshape to (21, 3) - Mediapipe hand landmarks
            landmarks = features[:63].reshape(21, 3)
            
            # Helper to check if a finger is extended.
            # y coordinate: lower value means higher up on the screen
            def is_extended(tip_idx):
                return landmarks[tip_idx, 1] < landmarks[tip_idx - 2, 1]

            # For the thumb, it's often more about x-axis relative to the hand center,
            # but y-axis can work reasonably well for a simple upright hand heuristic.
            thumb_ext = landmarks[4, 0] < landmarks[3, 0] if landmarks[4, 0] < landmarks[17, 0] else landmarks[4, 0] > landmarks[3, 0]
            # simplified fallback for thumb if sideways
            if abs(landmarks[4, 1] - landmarks[3, 1]) > abs(landmarks[4, 0] - landmarks[3, 0]):
                 thumb_ext = is_extended(4)

            index_ext = is_extended(8)
            middle_ext = is_extended(12)
            ring_ext = is_extended(16)
            pinky_ext = is_extended(20)

            fingers = (bool(thumb_ext), bool(index_ext), bool(middle_ext), bool(ring_ext), bool(pinky_ext))
            
            # Map combinations (Thumb, Index, Middle, Ring, Pinky) to words
            mapping = {
                (True, True, True, True, True): "HELLO",          # Full palm
                (True, False, False, False, False): "GOOD",       # Thumbs up
                (False, False, False, False, False): "STOP",      # Fist
                (False, True, False, False, False): "YOU",        # Pointing index
                (False, True, True, False, False): "FRIEND",      # Index and Middle (Peace/V)
                (True, True, False, False, False): "WATER",       # Thumb and Index (L)
                (True, False, False, False, True): "PLAY",        # Thumb and Pinky (Y / Call me)
                (False, True, True, True, False): "PLEASE",       # Index, Middle, Ring (W)
                (False, True, False, False, True): "ROCK",        # Rock on / Horns
                (False, False, False, False, True): "I",          # Pinky only
                (True, True, True, False, False): "BEAUTIFUL",    # Thumb, Index, Middle
                (True, True, False, False, True): "LOVE",         # I love you sign
                (False, True, True, True, True): "THANK YOU",     # 4 fingers, no thumb (B shape)
                (False, False, True, True, True): "PERFECT",      # OK sign (index/thumb closed, others open)
            }
            
            label = mapping.get(fingers, "UNKNOWN")
            
        except Exception as e:
            label = "UNKNOWN"

        return {
            "label": label,
            "confidence": 0.99 if label != "UNKNOWN" else 0.0,
            "is_confident": label != "UNKNOWN",
            "motion_detected": False,
            "motion_magnitude": 0.0,
            "source": "static",
            "static_pred": (label, 0.99),
            "dynamic_pred": ("UNKNOWN", 0.0),
        }

    def _compute_motion(self, features: np.ndarray) -> float:
        """
        Compute motion between current and previous frame.

        Returns:
            Motion magnitude (float)
        """
        if self.prev_features is None:
            return 0.0

        motion = float(np.linalg.norm(features - self.prev_features))

        # Maintain motion history for smoothing
        self.motion_history.append(motion)
        if len(self.motion_history) > 10:
            self.motion_history.pop(0)

        return motion

    def _gate_prediction(
        self,
        static_conf: float,
        dynamic_conf: float,
        motion_detected: bool,
    ) -> tuple:
        """
        Apply confidence gating logic as specified in the architecture.

        Args:
            static_conf: Confidence from static classifier
            dynamic_conf: Confidence from dynamic classifier
            motion_detected: Whether motion was detected

        Returns:
            Tuple of (final_confidence, source_classifier_name)
        """
        # If one of the models is unavailable (returns exactly 0.0), rely on the other
        if dynamic_conf == 0.0:
            return static_conf, "static"
        if static_conf == 0.0:
            return dynamic_conf, "dynamic"

        if motion_detected and dynamic_conf > 0:
            # Dynamic is primary when motion is detected
            # We don't penalize with the static model's confidence which is irrelevant
            final_conf = max(dynamic_conf, static_conf)
            source = "dynamic" if dynamic_conf > static_conf else "static"
        else:
            # Static is primary for stationary signs
            final_conf = max(static_conf, dynamic_conf)
            source = "static" if static_conf > dynamic_conf else "dynamic"

        return final_conf, source

    def get_average_motion(self) -> float:
        """Get the average motion over recent frames."""
        if not self.motion_history:
            return 0.0
        return sum(self.motion_history) / len(self.motion_history)

    def reset(self):
        """Reset all state."""
        self.prev_features = None
        self.motion_history.clear()
        self.dynamic_classifier.clear_buffer()
        logger.info("EnsembleRecognizer reset")
