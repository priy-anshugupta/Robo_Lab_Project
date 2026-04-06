"""
ISL-Multilingual Bridge — MediaPipe Landmark Extraction Module
Extracts and normalizes hand landmarks from video frames utilizing MediaPipe Tasks API.
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import logging
from typing import Optional
import os

from config.settings import (
    MEDIAPIPE_MIN_DETECTION_CONFIDENCE,
    MEDIAPIPE_MIN_TRACKING_CONFIDENCE,
    SINGLE_HAND_FEATURES,
)

logger = logging.getLogger("isl-bridge.vision")

# Hardcoded hand connections since mp.solutions is unavailable in Py3.13 pip builds
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index finger
    (5, 9), (9, 10), (10, 11), (11, 12),   # Middle finger
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring finger
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20) # Pinky
]

class LandmarkExtractor:
    """
    Wraps MediaPipe HandLandmarker Tasks API.
    Provides normalized feature vectors suitable for classification.
    """

    def __init__(
        self,
        min_detection_confidence: float = MEDIAPIPE_MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence: float = MEDIAPIPE_MIN_TRACKING_CONFIDENCE,
        static_image_mode: bool = False,
    ):
        model_path = os.path.join("models", "hand_landmarker.task")
        if not os.path.exists(model_path):
            logger.error(f"Missing MediaPipe model at {model_path}. Please download it.")
            raise FileNotFoundError(f"Missing {model_path}")
            
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_tracking_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.hand_landmarker = vision.HandLandmarker.create_from_options(options)

        logger.info(
            f"LandmarkExtractor initialized "
            f"(det_conf={min_detection_confidence}, track_conf={min_tracking_confidence})"
        )

    def extract(self, frame: np.ndarray) -> dict:
        """
        Extract all landmarks from a BGR frame.

        Args:
            frame: BGR numpy array from OpenCV

        Returns:
            dict with keys:
                - 'has_hands': bool — whether any hands were detected
                - 'left_hand': np.ndarray (21, 3) or None
                - 'right_hand': np.ndarray (21, 3) or None
                - 'pose': np.ndarray (33, 4) or None
                - 'face': np.ndarray (468, 3) or None (optional)
                - 'features': np.ndarray (63,) — normalized feature vector for dominant hand
                - 'features_both': np.ndarray (126,) or None — both hands feature vector
                - 'raw_results': MediaPipe results object
        """
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Process frame through MediaPipe
        results = self.hand_landmarker.detect(mp_image)

        output = {
            "has_hands": False,
            "left_hand": None,
            "right_hand": None,
            "pose": None,
            "face": None,
            "features": None,
            "features_both": None,
            "raw_results": results,
        }

        if results.handedness:
            output["has_hands"] = True
            for i, handedness in enumerate(results.handedness):
                # The model often swaps Left and Right, but we map them correctly
                label = handedness[0].category_name
                landmarks = results.hand_landmarks[i]
                
                lm_array = self._landmarks_to_array(landmarks, dims=3)
                
                if label == "Left":
                    output["left_hand"] = lm_array
                else:
                    output["right_hand"] = lm_array

            # Generate normalized feature vectors
            has_left = output["left_hand"] is not None
            has_right = output["right_hand"] is not None
            
            # Dominant hand features (prefer right hand)
            dominant = output["right_hand"] if has_right else output["left_hand"]
            output["features"] = self.normalize_landmarks(dominant)

            # Both hands features (if both detected)
            if has_left and has_right:
                left_norm = self.normalize_landmarks(output["left_hand"])
                right_norm = self.normalize_landmarks(output["right_hand"])
                output["features_both"] = np.concatenate([right_norm, left_norm])

        return output

    def _landmarks_to_array(self, landmarks, dims: int = 3) -> np.ndarray:
        """Convert MediaPipe landmark list to numpy array."""
        if dims == 4:
            return np.array([[lm.x, lm.y, lm.z, getattr(lm, 'visibility', 1.0)] for lm in landmarks])
        else:
            return np.array([[lm.x, lm.y, lm.z] for lm in landmarks])

    def normalize_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
        """Normalize hand landmarks relative to wrist anchor point."""
        if landmarks is None or len(landmarks) == 0:
            return np.zeros(SINGLE_HAND_FEATURES)

        # Step 1: Translate so wrist is at origin
        wrist = landmarks[0].copy()
        normalized = landmarks - wrist

        # Step 2: Scale by bounding box diagonal
        bbox_min = normalized.min(axis=0)
        bbox_max = normalized.max(axis=0)
        diagonal = np.linalg.norm(bbox_max - bbox_min)

        if diagonal > 1e-6:  # Avoid division by zero
            normalized = normalized / diagonal

        # Step 3: Flatten to 1D vector
        return normalized.flatten().astype(np.float32)

    def draw_landmarks(self, frame: np.ndarray, landmarks_dict: dict) -> np.ndarray:
        """Draw detected landmarks on the frame for visualization."""
        results = landmarks_dict.get("raw_results")
        if not results or not results.hand_landmarks:
            return frame

        annotated = frame.copy()
        h, w, _ = annotated.shape

        for i, hand_landmarks in enumerate(results.hand_landmarks):
            handedness = results.handedness[i][0].category_name
            color = (0, 255, 0) if handedness == "Left" else (255, 0, 0)
            
            # Draw connections
            for connection in HAND_CONNECTIONS:
                idx1, idx2 = connection
                lm1 = hand_landmarks[idx1]
                lm2 = hand_landmarks[idx2]
                pt1 = (int(lm1.x * w), int(lm1.y * h))
                pt2 = (int(lm2.x * w), int(lm2.y * h))
                cv2.line(annotated, pt1, pt2, color, 2)
            
            # Draw joints
            for lm in hand_landmarks:
                pt = (int(lm.x * w), int(lm.y * h))
                cv2.circle(annotated, pt, 4, color, -1)

        return annotated

    def compute_motion(
        self, prev_features: Optional[np.ndarray], curr_features: Optional[np.ndarray]
    ) -> float:
        """Compute motion magnitude between two consecutive frames."""
        if prev_features is None or curr_features is None:
            return 0.0
        return float(np.linalg.norm(curr_features - prev_features))

    def close(self):
        """Release MediaPipe resources."""
        if hasattr(self, 'hand_landmarker'):
            self.hand_landmarker.close()
        logger.info("LandmarkExtractor closed")

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
