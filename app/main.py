"""
ISL-Multilingual Bridge — Main Streamlit Application
Entry point for the ISL → Multilingual Text & Audio pipeline.

Launch with:
    streamlit run app/main.py
"""

import sys
from pathlib import Path
from datetime import datetime
import time

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

# ─── Page Configuration ─────────────────────────────────────────
st.set_page_config(
    page_title="ISL Multilingual Bridge",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "ISL-Multilingual Bridge — Real-time Indian Sign Language to Multilingual Text & Audio",
    },
)

# ─── Custom CSS ──────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global styles */
    .stApp {
        background: linear-gradient(135deg, #0F0F1A 0%, #1A1A2E 50%, #16213E 100%);
        font-family: 'Inter', sans-serif;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #12122A 0%, #0D0D1F 100%);
        border-right: 1px solid rgba(100, 100, 255, 0.1);
    }

    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #8888CC !important;
        font-size: 0.85rem !important;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-weight: 600;
    }

    /* Card-like containers */
    .stMarkdown div[style] {
        transition: all 0.3s ease;
    }

    /* Metric styling */
    [data-testid="stMetric"] {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(100,100,200,0.15);
        border-radius: 10px;
        padding: 0.5rem;
    }

    [data-testid="stMetricValue"] {
        color: #80B0FF !important;
        font-weight: 700 !important;
    }

    [data-testid="stMetricLabel"] {
        color: #888 !important;
    }

    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #4A2FBD, #2E86DE) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(74, 47, 189, 0.3) !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(74, 47, 189, 0.5) !important;
    }

    /* Select box */
    .stSelectbox > div > div {
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid rgba(100,100,200,0.2) !important;
        border-radius: 8px !important;
    }

    /* Slider */
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #4A2FBD, #2E86DE) !important;
    }

    /* Divider */
    hr {
        border-color: rgba(100,100,200,0.1) !important;
    }

    /* Toast */
    .stToast {
        background: rgba(30,30,50,0.95) !important;
        border: 1px solid rgba(100,100,255,0.2) !important;
    }

    /* Header styling */
    .main-header {
        text-align: center;
        padding: 1.5rem 0 1rem;
        margin-bottom: 1rem;
    }
    .main-header h1 {
        background: linear-gradient(135deg, #667EEA, #764BA2, #F093FB);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
    }
    .main-header p {
        color: #888;
        font-size: 0.9rem;
        margin-top: 0.3rem;
    }

    /* Demo mode banner */
    .demo-banner {
        background: linear-gradient(135deg, rgba(255,165,0,0.15), rgba(255,100,0,0.1));
        border: 1px solid rgba(255,165,0,0.3);
        border-radius: 10px;
        padding: 0.8rem 1rem;
        margin-bottom: 1rem;
        text-align: center;
    }
    .demo-banner p {
        color: #FFB347;
        margin: 0;
        font-size: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ─── Initialize Session State ───────────────────────────────────
def init_session_state():
    """Initialize all session state variables with defaults."""
    defaults = {
        "running": False,
        "sign_buffer": [],
        "english_output": "",
        "translated_output": "",
        "audio_bytes": None,
        "history": [],
        "target_language": "Hindi",
        "confidence_threshold": 0.80,
        "pause_timeout": 1.5,
        "tts_backend": "gtts",
        "auto_play": True,
        "total_signs": 0,
        "manual_submit": False,
        "latest_prediction": None,
        "webrtc_playing": False,
        "webrtc_started_at": 0.0,
        "_ui_refresh_ts": 0.0,
        "candidate_label": "",
        "candidate_count": 0,
        "last_accept_ts": 0.0,
        "demo_mode": True,  # True until models are trained
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()


# ─── Initialize Pipeline Components ─────────────────────────────
@st.cache_resource
def load_pipeline():
    """Load all ML pipeline components (cached)."""
    from config.settings import STATIC_MODEL_PATH, DYNAMIC_MODEL_PATH

    components = {
        "landmark_extractor": None,
        "recognizer": None,
        "sentence_builder": None,
        "translator": None,
        "tts_engine": None,
        "models_available": False,
    }

    try:
        from vision.landmark_extractor import LandmarkExtractor
        components["landmark_extractor"] = LandmarkExtractor()
    except Exception as e:
        st.warning(f"⚠ Vision module unavailable: {e}")

    try:
        from recognition.ensemble import EnsembleRecognizer
        recognizer = EnsembleRecognizer()
        # Check if actual models are loaded
        if recognizer.static_classifier.is_loaded:
            components["recognizer"] = recognizer
            components["models_available"] = True
    except Exception as e:
        pass  # Models not trained yet

    try:
        from nlp.gpt_client import GPTClient
        from nlp.sentence_builder import SentenceBuilder
        from nlp.translator import Translator

        gpt = GPTClient()
        components["sentence_builder"] = SentenceBuilder(gpt)
        components["translator"] = Translator(gpt)
    except Exception as e:
        st.warning(f"⚠ NLP module unavailable: {e}")

    try:
        from audio.tts_engine import TTSEngine
        components["tts_engine"] = TTSEngine()
    except Exception as e:
        st.warning(f"⚠ Audio module unavailable: {e}")

    return components


pipeline = load_pipeline()


# ─── Render UI ──────────────────────────────────────────────────
from app.components.sidebar import render_sidebar
from app.components.video_panel import render_video_panel, drain_predictions
from app.components.output_panel import render_output_panel
from app.components.history_panel import render_history_panel


def ingest_live_predictions():
    """Consume queued live predictions and update UI-facing session state."""
    predictions = drain_predictions(max_items=96)
    if not predictions:
        return

    conf_threshold = float(st.session_state.get("confidence_threshold", 0.8) or 0.8)
    sign_buffer = st.session_state.get("sign_buffer", [])
    last_added_sign = st.session_state.get("last_added_sign")
    last_added_ts = float(st.session_state.get("last_added_sign_ts", 0.0) or 0.0)
    candidate_label = str(st.session_state.get("candidate_label", "") or "")
    candidate_count = int(st.session_state.get("candidate_count", 0) or 0)
    last_accept_ts = float(st.session_state.get("last_accept_ts", 0.0) or 0.0)

    # Small stabilization delay to avoid jittery sign spam.
    min_accept_gap_sec = 0.25
    min_consecutive_frames = 2

    for pred in predictions:
        st.session_state["latest_prediction"] = pred

        is_confident = bool(pred.get("is_confident", False))
        confidence = float(pred.get("confidence", 0.0) or 0.0)
        label = str(pred.get("label", "") or "").strip().upper()
        pred_ts = float(pred.get("ts", time.time()) or time.time())

        if not is_confident or not label or confidence < conf_threshold:
            candidate_label = ""
            candidate_count = 0
            continue

        if label == candidate_label:
            candidate_count += 1
        else:
            candidate_label = label
            candidate_count = 1

        if candidate_count < min_consecutive_frames:
            continue

        if pred_ts - last_accept_ts < min_accept_gap_sec:
            continue

        # Debounce repeated signs while keeping intentional repeats possible.
        if (label != last_added_sign) or (pred_ts - last_added_ts >= 0.8):
            sign_buffer.append(label)
            sign_buffer = sign_buffer[-20:]
            last_added_sign = label
            last_added_ts = pred_ts
            last_accept_ts = pred_ts

    st.session_state["sign_buffer"] = sign_buffer
    st.session_state["last_added_sign"] = last_added_sign
    st.session_state["last_added_sign_ts"] = last_added_ts
    st.session_state["candidate_label"] = candidate_label
    st.session_state["candidate_count"] = candidate_count
    st.session_state["last_accept_ts"] = last_accept_ts

# Sidebar
render_sidebar()

# ─── Main Content ────────────────────────────────────────────────

# Header
st.markdown(
    """
    <div class="main-header">
        <h1>🤟 ISL Multilingual Bridge</h1>
        <p>Real-time Indian Sign Language → Text → Translation → Audio</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Demo mode notice (when models aren't trained)
if not pipeline.get("models_available"):
    st.markdown(
        """
        <div class="demo-banner">
            <p>🧪 <strong>Demo Mode</strong> — ML models not yet trained.
            Run <code>python training/data_collector.py</code> to collect data,
            then <code>python training/train_static.py</code> to train models.
            The app interface is fully functional for testing.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Main layout: 2 columns
col_left, col_right = st.columns([1.2, 1])

with col_left:
    # Video panel
    render_video_panel(
        landmark_extractor=pipeline.get("landmark_extractor"),
        sign_recognizer=pipeline.get("recognizer"),
    )

    # Pull live predictions recognized by the video thread into UI state.
    ingest_live_predictions()

    # Demo controls (when no real recognition is available)
    if not pipeline.get("models_available"):
        st.markdown("---")
        st.markdown("#### 🧪 Demo Input")
        st.markdown(
            '<p style="color: #888; font-size: 0.8rem;">'
            "Type ISL gloss words to test the NLP + TTS pipeline:</p>",
            unsafe_allow_html=True,
        )
        demo_input = st.text_input(
            "ISL Gloss (e.g., HELLO MY NAME PRIYA)",
            key="demo_gloss",
            placeholder="HELLO MY NAME PRIYA",
        )

        if st.button("🚀 Process Demo Input", key="demo_submit", use_container_width=True):
            if demo_input:
                signs = demo_input.strip().upper().split()
                st.session_state["sign_buffer"] = signs
                st.session_state["manual_submit"] = True

with col_right:
    # Output panel
    render_output_panel()

# ─── Process Manual Submit ──────────────────────────────────────
if st.session_state.get("manual_submit", False):
    st.session_state["manual_submit"] = False
    signs = st.session_state.get("sign_buffer", [])

    if signs:
        with st.spinner("🔄 Processing sentence..."):
            # Stage 1: ISL → English
            if pipeline.get("sentence_builder"):
                result = pipeline["sentence_builder"].build(signs)
                english = result.get("english", " ".join(signs))
            else:
                english = " ".join(w.capitalize() for w in signs) + "."

            st.session_state["english_output"] = english

            # Stage 2: English → Target Language
            target_lang = st.session_state.get("target_language", "Hindi")
            if pipeline.get("translator") and target_lang != "English":
                trans_result = pipeline["translator"].translate(english, target_lang)
                translated = trans_result.get("translated", english)
            else:
                translated = english

            st.session_state["translated_output"] = translated

            # Stage 3: Text → Audio
            tts_text = translated if target_lang != "English" else english
            if pipeline.get("tts_engine"):
                audio_bytes = pipeline["tts_engine"].synthesize(tts_text, target_lang)
                st.session_state["audio_bytes"] = audio_bytes

            # Add to history
            history_entry = {
                "gloss": " ".join(signs),
                "english": english,
                "translated": translated,
                "language": target_lang,
                "timestamp": datetime.now().strftime("%H:%M:%S"),
            }
            st.session_state["history"].append(history_entry)
            st.session_state["total_signs"] = st.session_state.get("total_signs", 0) + len(signs)

            # Clear buffer after submission
            st.session_state["sign_buffer"] = []

        st.rerun()

# Keep UI live while camera is running, without requiring manual page refresh.
if st.session_state.get("webrtc_playing", False):
    started_at = float(st.session_state.get("webrtc_started_at", 0.0) or 0.0)
    now = time.time()
    # Give WebRTC a short warm-up period so camera startup is not interrupted.
    if started_at > 0 and (now - started_at) >= 2.0:
        last_refresh = float(st.session_state.get("_ui_refresh_ts", 0.0) or 0.0)
        if now - last_refresh >= 0.7:
            st.session_state["_ui_refresh_ts"] = now
            st.rerun()

# ─── History Panel ──────────────────────────────────────────────
st.markdown("---")
render_history_panel()

# ─── Footer ─────────────────────────────────────────────────────
st.markdown(
    """
    <div style="
        text-align: center;
        padding: 2rem 0 1rem;
        color: #555;
        font-size: 0.75rem;
    ">
        <p>ISL-Multilingual Bridge v1.0.0 | Powered by MediaPipe + GPT-4.1 Mini + gTTS</p>
        <p>Made with ❤️ for the deaf and hard-of-hearing community</p>
    </div>
    """,
    unsafe_allow_html=True,
)
