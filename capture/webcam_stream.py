"""
ISL-Multilingual Bridge — Webcam Capture & Streaming Module
Handles OpenCV video capture and WebRTC integration for Streamlit.
"""

import cv2
import numpy as np
import logging
from threading import Thread, Lock
from typing import Optional

logger = logging.getLogger("isl-bridge.capture")


class WebcamStream:
    """
    Manages OpenCV VideoCapture with threaded frame buffering
    for smooth, non-blocking webcam access.
    """

    def __init__(self, src: int = 0, width: int = 640, height: int = 480):
        """
        Initialize webcam stream.

        Args:
            src: Camera index (0 = default webcam)
            width: Desired frame width
            height: Desired frame height
        """
        self.src = src
        self.width = width
        self.height = height
        self.cap: Optional[cv2.VideoCapture] = None
        self.frame: Optional[np.ndarray] = None
        self.running = False
        self._lock = Lock()
        self._thread: Optional[Thread] = None

        logger.info(f"WebcamStream initialized (src={src}, {width}x{height})")

    def start(self) -> "WebcamStream":
        """Start the webcam capture thread."""
        if self.running:
            logger.warning("WebcamStream already running")
            return self

        self.cap = cv2.VideoCapture(self.src)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {self.src}")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        self.running = True
        self._thread = Thread(target=self._update, daemon=True)
        self._thread.start()
        logger.info("WebcamStream started")
        return self

    def _update(self):
        """Continuously read frames in background thread."""
        while self.running:
            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    with self._lock:
                        self.frame = frame
                else:
                    logger.warning("Failed to read frame from camera")

    def get_frame(self) -> Optional[np.ndarray]:
        """
        Get the latest frame from the buffer.

        Returns:
            BGR numpy array or None if no frame available
        """
        with self._lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        """Stop the webcam capture."""
        self.running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self.cap:
            self.cap.release()
        self.frame = None
        logger.info("WebcamStream stopped")

    def __del__(self):
        self.stop()


def preprocess_frame(frame: np.ndarray, target_size: tuple = (640, 480)) -> np.ndarray:
    """
    Preprocess a raw frame for MediaPipe processing.

    Args:
        frame: Raw BGR frame from OpenCV
        target_size: (width, height) to resize to

    Returns:
        Preprocessed BGR frame (flipped horizontally for mirror effect)
    """
    # Flip horizontally for a mirror / selfie view
    frame = cv2.flip(frame, 1)

    # Resize if needed
    h, w = frame.shape[:2]
    if (w, h) != target_size:
        frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)

    return frame


class WebRTCProcessor:
    """
    Video processor for streamlit-webrtc integration.
    Processes each incoming video frame through the ISL pipeline.
    """

    def __init__(self, landmark_extractor=None, sign_recognizer=None):
        """
        Args:
            landmark_extractor: LandmarkExtractor instance for hand detection
            sign_recognizer: Ensemble classifier for sign recognition
        """
        self.landmark_extractor = landmark_extractor
        self.sign_recognizer = sign_recognizer
        self.latest_landmarks = None
        self.latest_prediction = None
        self._lock = Lock()

    def recv(self, frame):
        """
        Process a single video frame from WebRTC.
        This is called by streamlit-webrtc for each frame.

        Args:
            frame: av.VideoFrame from WebRTC

        Returns:
            Processed av.VideoFrame with landmark overlay
        """
        try:
            import av
        except ImportError:
            logger.error("PyAV not installed. Install with: pip install av")
            return frame

        # Convert av.VideoFrame to numpy array
        img = frame.to_ndarray(format="bgr24")

        # Mirror the frame
        img = cv2.flip(img, 1)

        # Extract and draw landmarks if extractor is available
        if self.landmark_extractor:
            landmarks = self.landmark_extractor.extract(img)
            with self._lock:
                self.latest_landmarks = landmarks

            # Draw landmark overlay
            img = self.landmark_extractor.draw_landmarks(img, landmarks)

            # Run sign recognition if recognizer is available
            if self.sign_recognizer and landmarks.get("has_hands"):
                features = landmarks.get("features")
                if features is not None:
                    prediction = self.sign_recognizer.predict(features)
                    with self._lock:
                        self.latest_prediction = prediction

                    # Draw prediction label on frame
                    if prediction and prediction.get("is_confident"):
                        label = prediction["label"]
                        conf = prediction["confidence"]
                        cv2.putText(
                            img,
                            f"{label} ({conf:.2f})",
                            (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.2,
                            (0, 255, 0),
                            3,
                        )

        # Convert back to av.VideoFrame
        return av.VideoFrame.from_ndarray(img, format="bgr24")

    def get_latest_prediction(self) -> Optional[dict]:
        """Get the most recent sign prediction."""
        with self._lock:
            return self.latest_prediction

    def get_latest_landmarks(self) -> Optional[dict]:
        """Get the most recent landmark data."""
        with self._lock:
            return self.latest_landmarks
