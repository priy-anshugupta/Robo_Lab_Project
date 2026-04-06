"""
ISL-Multilingual Bridge — Video Panel Component
Webcam feed display with landmark overlay using streamlit-webrtc.
"""

import streamlit as st
import logging
import time
from collections import deque
from threading import Lock

logger = logging.getLogger("isl-bridge.app.video")

_PREDICTION_QUEUE = deque(maxlen=512)
_QUEUE_LOCK = Lock()


def drain_predictions(max_items: int = 64):
    """Drain queued predictions from the video thread."""
    items = []
    with _QUEUE_LOCK:
        while _PREDICTION_QUEUE and len(items) < max_items:
            items.append(_PREDICTION_QUEUE.popleft())
    return items


def render_video_panel(landmark_extractor=None, sign_recognizer=None):
    """
    Render the webcam video panel with landmark overlay.

    Uses streamlit-webrtc for browser-based camera access.
    Falls back to st.camera_input() if WebRTC is not available.

    Args:
        landmark_extractor: LandmarkExtractor instance
        sign_recognizer: EnsembleRecognizer instance
    """
    st.markdown(
        """
        <div style="
            background: linear-gradient(135deg, rgba(30,30,60,0.8), rgba(20,20,40,0.9));
            border: 1px solid rgba(100,100,255,0.2);
            border-radius: 12px;
            padding: 1rem;
            margin-bottom: 1rem;
        ">
            <h4 style="margin: 0 0 0.5rem 0; color: #A0A0FF;">📹 Live Camera Feed</h4>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        _render_webrtc(landmark_extractor, sign_recognizer)
    except Exception as e:
        logger.warning(f"WebRTC failed: {e}. Using fallback camera.")
        _render_fallback_camera(landmark_extractor)


def _render_webrtc(landmark_extractor, sign_recognizer):
    """Render webcam using streamlit-webrtc."""
    from streamlit_webrtc import webrtc_streamer, WebRtcMode
    import av
    import cv2
    import numpy as np

    class ISLVideoProcessor:
        """Custom video processor for ISL recognition pipeline."""

        def __init__(self):
            self.extractor = landmark_extractor
            self.recognizer = sign_recognizer
            self.latest_prediction = None

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            img = cv2.flip(img, 1)

            if self.extractor:
                landmarks = self.extractor.extract(img)
                img = self.extractor.draw_landmarks(img, landmarks)

                if self.recognizer and landmarks.get("has_hands"):
                    features = landmarks.get("features")
                    if features is not None:
                        prediction = self.recognizer.predict(features)
                        self.latest_prediction = prediction

                        if prediction:
                            with _QUEUE_LOCK:
                                _PREDICTION_QUEUE.append(
                                    {
                                        "label": prediction.get("label"),
                                        "confidence": prediction.get("confidence", 0.0),
                                        "is_confident": prediction.get("is_confident", False),
                                        "ts": time.time(),
                                    }
                                )

                        if prediction.get("is_confident"):
                            label = prediction["label"]
                            conf = prediction["confidence"]

                            # Draw prediction label with background
                            text = f"{label} ({conf:.0%})"
                            (tw, th), _ = cv2.getTextSize(
                                text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2
                            )
                            cv2.rectangle(
                                img, (8, 8), (18 + tw, 48 + th),
                                (0, 0, 0), -1
                            )
                            cv2.putText(
                                img, text, (12, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                                (0, 255, 128), 2,
                            )

                # Hand detection status indicator
                has_hands = landmarks.get("has_hands", False)
                status = "Hand Detected" if has_hands else "No Hand"
                color = (0, 255, 0) if has_hands else (0, 0, 255)
                cv2.putText(
                    img, status, (10, img.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
                )

            return av.VideoFrame.from_ndarray(img, format="bgr24")

    # WebRTC streamer configuration
    ctx = webrtc_streamer(
        key="isl-webcam",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=ISLVideoProcessor,
        media_stream_constraints={
            "video": {
                "width": {"ideal": 640},
                "height": {"ideal": 480},
                "frameRate": {"ideal": 30},
            },
            "audio": False,
        },
        async_processing=True,
    )

    if getattr(ctx, "state", None):
        is_playing = bool(ctx.state.playing)
        st.session_state["running"] = is_playing
        st.session_state["webrtc_playing"] = is_playing
        if is_playing and not st.session_state.get("webrtc_started_at"):
            st.session_state["webrtc_started_at"] = time.time()
        if not is_playing:
            st.session_state["webrtc_started_at"] = 0.0

    # Keep latest prediction visible even if no new UI interaction occurs.
    if ctx.video_processor and ctx.video_processor.latest_prediction:
        st.session_state["latest_prediction"] = ctx.video_processor.latest_prediction


def _render_fallback_camera(landmark_extractor):
    """Fallback camera using Streamlit's built-in camera input."""
    st.info("💡 WebRTC not available. Using snapshot mode instead.")

    camera_input = st.camera_input("Capture a sign", key="camera_fallback")

    if camera_input is not None:
        import cv2
        import numpy as np
        from PIL import Image

        # Convert to OpenCV format
        image = Image.open(camera_input)
        frame = np.array(image)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if landmark_extractor:
            landmarks = landmark_extractor.extract(frame)
            annotated = landmark_extractor.draw_landmarks(frame, landmarks)

            # Convert back to RGB for display
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            st.image(annotated_rgb, caption="Detected landmarks", use_container_width=True)

            if landmarks.get("has_hands") and landmarks.get("features") is not None:
                st.success("✋ Hand detected!")
                st.session_state["latest_features"] = landmarks["features"]
            else:
                st.warning("No hand detected. Please show your hand clearly.")
        else:
            st.image(camera_input, caption="Camera feed", use_container_width=True)
