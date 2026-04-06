"""
ISL-Multilingual Bridge — Configuration Settings
Loads from .env file with sensible defaults.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# ─── OpenAI Configuration ─────────────────────────────────────
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

# ─── Recognition Configuration ────────────────────────────────
CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.80"))
FRAME_SEQUENCE_LENGTH: int = int(os.getenv("FRAME_SEQUENCE_LENGTH", "30"))
SENTENCE_PAUSE_TIMEOUT: float = float(os.getenv("SENTENCE_PAUSE_TIMEOUT", "1.5"))

# ─── Language Configuration ───────────────────────────────────
SUPPORTED_LANGUAGES: list = ["Hindi", "Tamil", "Bengali", "Telugu", "Marathi"]

LANGUAGE_CODES: dict = {
    "Hindi": "hi",
    "Tamil": "ta",
    "Bengali": "bn",
    "Telugu": "te",
    "Marathi": "mr",
    "English": "en",
}

# ─── TTS Configuration ────────────────────────────────────────
TTS_BACKEND: str = os.getenv("TTS_BACKEND", "gtts")  # "gtts" or "pyttsx3"

# ─── Model Paths ──────────────────────────────────────────────
MODELS_DIR = PROJECT_ROOT / "models"
STATIC_MODEL_PATH = MODELS_DIR / "static_classifier.pkl"
DYNAMIC_MODEL_PATH = MODELS_DIR / "dynamic_classifier.h5"
STATIC_LABEL_ENCODER_PATH = MODELS_DIR / "static_label_encoder.pkl"
DYNAMIC_LABEL_ENCODER_PATH = MODELS_DIR / "dynamic_label_encoder.pkl"

# ─── Data Paths ───────────────────────────────────────────────
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SIGN_LIST_PATH = DATA_DIR / "isl_sign_list.json"

# ─── MediaPipe Configuration ──────────────────────────────────
MEDIAPIPE_MIN_DETECTION_CONFIDENCE: float = 0.7
MEDIAPIPE_MIN_TRACKING_CONFIDENCE: float = 0.5

# ─── Feature Dimensions ──────────────────────────────────────
SINGLE_HAND_FEATURES: int = 63   # 21 landmarks × 3 coords
BOTH_HANDS_FEATURES: int = 126   # 42 landmarks × 3 coords
HAND_POSE_FEATURES: int = 99     # 63 hand + 33 pose (x,y) + 3 extra

# ─── Logging ──────────────────────────────────────────────────
import logging

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger("isl-bridge")
