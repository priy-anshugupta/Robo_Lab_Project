# 🤟 Indian Sign Language (ISL) → Multilingual Text & Audio Output
## Complete Implementation Plan

---

## 📋 Table of Contents

1. [Project Overview](#1-project-overview)
2. [Goals & Success Metrics](#2-goals--success-metrics)
3. [System Architecture](#3-system-architecture)
4. [Technology Stack](#4-technology-stack)
5. [Data Flow Diagram](#5-data-flow-diagram)
6. [Module Breakdown](#6-module-breakdown)
7. [AI/ML Pipeline Design](#7-aiml-pipeline-design)
8. [GPT-4.1 Mini Integration](#8-gpt-41-mini-integration)
9. [Multilingual Support Plan](#9-multilingual-support-plan)
10. [Streamlit App Design](#10-streamlit-app-design)
11. [Directory Structure](#11-directory-structure)
12. [Phase-wise Implementation Roadmap](#12-phase-wise-implementation-roadmap)
13. [API & Prompt Engineering](#13-api--prompt-engineering)
14. [Dataset Strategy](#14-dataset-strategy)
15. [Testing & Evaluation Plan](#15-testing--evaluation-plan)
16. [Deployment Plan](#16-deployment-plan)
17. [Risk Register](#17-risk-register)
18. [Future Roadmap](#18-future-roadmap)

---

## 1. Project Overview

**Project Name:** ISL-Multilingual Bridge  
**Version:** 1.0.0  
**Domain:** Assistive Technology / Sign Language Recognition / NLP  
**Primary Goal:** Enable deaf and hard-of-hearing individuals to communicate in real-time by converting Indian Sign Language (ISL) gestures captured via webcam into text and audio in multiple Indian and global languages.

### Problem Statement

Over 18 million people in India are deaf or hard of hearing. Indian Sign Language (ISL) is the primary mode of communication for the deaf community, yet non-signers — including doctors, teachers, employers, and family members — cannot understand it. This creates a massive communication barrier in daily life, healthcare, education, and employment.

### Solution

A real-time, AI-powered pipeline that:
- Captures webcam video of ISL gestures
- Detects hand landmarks and interprets signs
- Converts recognized signs into English text using a trained classifier
- Uses GPT-4.1 Mini to refine, contextualize, and translate the text into 5 languages
- Outputs both text and audio (text-to-speech) in the selected language
- Delivered via an accessible, intuitive **Streamlit** web application

---

## 2. Goals & Success Metrics

| Goal | Metric | Target |
|------|--------|--------|
| Real-time gesture recognition | Inference latency | < 300ms per frame |
| Sign classification accuracy | Top-1 accuracy | ≥ 85% on test set |
| Multilingual translation quality | BLEU Score | ≥ 0.65 |
| Audio output naturalness | MOS Score (user study) | ≥ 4.0 / 5.0 |
| App usability | System Usability Scale | ≥ 75 |
| Supported signs (MVP) | Number of distinct ISL signs | 100+ signs |
| Language support | Number of languages | 5 languages |

---

## 3. System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        STREAMLIT FRONTEND                           │
│   ┌────────────┐  ┌──────────────┐  ┌───────────┐  ┌────────────┐  │
│   │  Webcam    │  │  Language    │  │   Text    │  │  Audio     │  │
│   │  Feed      │  │  Selector    │  │  Output   │  │  Player    │  │
│   └────────────┘  └──────────────┘  └───────────┘  └────────────┘  │
└──────────────────────────┬──────────────────────────────────────────┘
                           │ Video Frames (OpenCV)
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    VISION PROCESSING LAYER                          │
│                                                                     │
│   ┌──────────────────────────────────────────────────────────────┐  │
│   │             MediaPipe Holistic / Hands                       │  │
│   │  • 21 hand landmarks (x, y, z) per hand                     │  │
│   │  • 33 pose landmarks (optional for full-body ISL)            │  │
│   │  • Face mesh landmarks (for facial grammar cues)             │  │
│   └──────────────────────────────────────────────────────────────┘  │
│                           │                                         │
│                           ▼                                         │
│   ┌──────────────────────────────────────────────────────────────┐  │
│   │              Feature Extraction                              │  │
│   │  • Landmark normalization (relative to wrist anchor)         │  │
│   │  • Velocity & direction vectors (temporal features)          │  │
│   │  • Handshape encoding                                        │  │
│   └──────────────────────────────────────────────────────────────┘  │
└──────────────────────────┬──────────────────────────────────────────┘
                           │ Feature Vectors
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    GESTURE RECOGNITION ENGINE                       │
│                                                                     │
│   ┌────────────────────┐    ┌──────────────────────────────────┐   │
│   │  Static Sign       │    │  Dynamic Sign Classifier         │   │
│   │  Classifier        │    │  (LSTM / Transformer)            │   │
│   │  (Random Forest /  │    │  • Sequence of frames (30fps)    │   │
│   │   MLP / CNN)       │    │  • Motion trajectory             │   │
│   └────────────────────┘    └──────────────────────────────────┘   │
│              │                              │                       │
│              └──────────┬───────────────────┘                       │
│                         ▼                                           │
│              ┌────────────────────┐                                 │
│              │  Ensemble Decoder  │                                 │
│              │  + Confidence Gate │                                 │
│              └────────────────────┘                                 │
└──────────────────────────┬──────────────────────────────────────────┘
                           │ Recognized Word/Sign (English)
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    GPT-4.1 MINI INTEGRATION LAYER                   │
│                                                                     │
│   ┌──────────────────────────────────────────────────────────────┐  │
│   │                  Sign Stream Buffer                          │  │
│   │  ["HELLO", "MY", "NAME", "PRIYA"] → context window          │  │
│   └──────────────────────────────────────────────────────────────┘  │
│                           │                                         │
│                           ▼                                         │
│   ┌──────────────────────────────────────────────────────────────┐  │
│   │               OpenAI GPT-4.1 Mini API                        │  │
│   │  Task 1: Sentence Formation (ISL → Grammatical English)      │  │
│   │  Task 2: Translation to target language                      │  │
│   │  Task 3: Context enrichment & ambiguity resolution           │  │
│   └──────────────────────────────────────────────────────────────┘  │
└──────────────────────────┬──────────────────────────────────────────┘
                           │ Translated Text
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    AUDIO SYNTHESIS LAYER                            │
│                                                                     │
│   ┌──────────────────────────────────────────────────────────────┐  │
│   │              Text-to-Speech Engine                           │  │
│   │  • gTTS (Google TTS) — Primary                              │  │
│   │  • pyttsx3 — Offline fallback                                │  │
│   │  • Language-specific voice model selection                   │  │
│   └──────────────────────────────────────────────────────────────┘  │
└──────────────────────────┬──────────────────────────────────────────┘
                           │ Audio (.mp3 / .wav)
                           ▼
                  Streamlit Audio Player
```

---

## 4. Technology Stack

### Core ML & Vision

| Component | Technology | Version |
|-----------|-----------|---------|
| Hand/Pose Landmark Detection | MediaPipe Holistic | ≥ 0.10 |
| Image Processing | OpenCV (cv2) | ≥ 4.8 |
| Static Sign Classifier | Scikit-learn (Random Forest + MLP) | ≥ 1.3 |
| Dynamic Sign Classifier | TensorFlow/Keras LSTM | ≥ 2.13 |
| Feature Engineering | NumPy, SciPy | Latest |
| Model Serialization | Pickle, HDF5 (Keras) | — |

### AI / LLM

| Component | Technology |
|-----------|-----------|
| NLP & Translation | OpenAI GPT-4.1 Mini API |
| Prompt Management | Custom prompt templates (Python) |
| Sentence Refinement | GPT-4.1 Mini (grammar correction) |
| Multilingual Translation | GPT-4.1 Mini (zero-shot) |

### Audio

| Component | Technology |
|-----------|-----------|
| Text-to-Speech (Online) | gTTS (Google Text-to-Speech) |
| Text-to-Speech (Offline) | pyttsx3 |
| Audio Playback | Streamlit `st.audio()` |

### Frontend / App

| Component | Technology |
|-----------|-----------|
| Web Application | Streamlit |
| Webcam Capture | streamlit-webrtc + OpenCV |
| Real-time Streaming | WebRTC (aiortc) |
| UI Components | Streamlit native + custom CSS |

### Backend / Infra

| Component | Technology |
|-----------|-----------|
| Language | Python 3.10+ |
| Config Management | python-dotenv |
| Logging | Python logging module |
| Environment | Conda / venv |
| Containerization | Docker (optional for deployment) |

---

## 5. Data Flow Diagram

```
USER (shows sign)
      │
      ▼
[Webcam / Camera Feed]
      │
      ▼ (raw frames @ 30fps)
[OpenCV Frame Capture]
      │
      ▼ (BGR frames)
[MediaPipe Holistic Processing]
      │
      ├─ Hand Landmarks (42 points: 21 per hand × x,y,z)
      ├─ Pose Landmarks (33 body keypoints)
      └─ Face Landmarks (468 mesh points, optional)
      │
      ▼
[Feature Normalization & Extraction]
      │ (normalized 63-dim vector per frame)
      ▼
   ┌─────────────────────────────┐
   │   Static?  ──Yes──►  MLP / Random Forest Classifier
   │   or                        │
   │   Dynamic? ──Yes──►  LSTM Sequence Classifier (30 frames)
   └─────────────────────────────┘
      │
      ▼ (predicted ISL word, confidence score)
[Confidence Gate: threshold > 0.80?]
      │
   Yes│                    No │
      ▼                       ▼
[Add to Sign Buffer]    [Discard / Wait]
      │
      ▼ (buffer: ["HELLO", "MY", "NAME", ...])
[Sentence Trigger] ← (pause detection / manual submit)
      │
      ▼
[GPT-4.1 Mini - Stage 1]
  Prompt: "Convert ISL gloss to grammatical English: {signs}"
      │
      ▼ (English sentence)
[GPT-4.1 Mini - Stage 2]
  Prompt: "Translate to {target_language}: {english_sentence}"
      │
      ▼ (translated text)
[gTTS Audio Synthesis]
  → audio bytes (mp3)
      │
      ▼
[Streamlit UI Output]
  ├─ English Text Display
  ├─ Translated Text Display
  └─ Audio Player (auto-play)
```

---

## 6. Module Breakdown

### Module 1: `capture/` — Video Capture & Streaming

**File:** `capture/webcam_stream.py`  
**Responsibility:** Capture live video from webcam, handle WebRTC for Streamlit  

Key classes:
- `WebcamStream` — Manages OpenCV VideoCapture, frame buffering
- `WebRTCProcessor` — Inherits `VideoProcessorBase` for streamlit-webrtc integration

Key functions:
- `get_frame()` → Returns BGR numpy array
- `preprocess_frame(frame)` → Resize, flip, normalize for MediaPipe

---

### Module 2: `vision/` — MediaPipe Landmark Extraction

**File:** `vision/landmark_extractor.py`  
**Responsibility:** Run MediaPipe Holistic on each frame, extract coordinates

Key classes:
- `LandmarkExtractor` — Wraps MediaPipe Holistic pipeline

Key functions:
- `extract(frame)` → Returns dict of `{hands, pose, face}` landmark arrays
- `normalize_landmarks(landmarks)` → Normalizes relative to wrist anchor point
- `draw_landmarks(frame, landmarks)` → Visual overlay for debugging

Output shape: `np.array of shape (63,)` for hand features, `(99,)` for hand+pose

---

### Module 3: `recognition/` — Sign Classification

**File:** `recognition/static_classifier.py`  
**Responsibility:** Classify static (non-moving) ISL signs (alphabet, static words)

- Model: Random Forest or MLP (scikit-learn)
- Input: Single frame landmark vector `(63,)`
- Output: `(label: str, confidence: float)`

**File:** `recognition/dynamic_classifier.py`  
**Responsibility:** Classify dynamic ISL signs (words involving motion)

- Model: Bidirectional LSTM (TensorFlow/Keras)
- Input: Sequence of 30 frames `(30, 63)`
- Output: `(label: str, confidence: float)`

**File:** `recognition/ensemble.py`  
**Responsibility:** Combine static + dynamic predictions, apply confidence gate

- Confidence threshold: configurable (default: 0.80)
- Tie-breaking: prefer dynamic classifier if motion detected
- Output: `(final_label: str, confidence: float, is_confident: bool)`

**File:** `recognition/sign_buffer.py`  
**Responsibility:** Accumulate recognized signs into a sentence buffer

- Buffer: `deque(maxlen=20)` of sign labels
- Sentence trigger: pause > 1.5 seconds OR manual submit
- Deduplication: collapse consecutive identical signs

---

### Module 4: `nlp/` — GPT-4.1 Mini Integration

**File:** `nlp/gpt_client.py`  
**Responsibility:** Wrapper for OpenAI API calls with retry logic

**File:** `nlp/sentence_builder.py`  
**Responsibility:** Convert ISL gloss word stream → grammatical English sentence

**File:** `nlp/translator.py`  
**Responsibility:** Translate English sentence to target language via GPT-4.1 Mini

Supported languages:
- Hindi (`hi`)
- Tamil (`ta`)
- Bengali (`bn`)
- Telugu (`te`)
- Marathi (`mr`)

**File:** `nlp/prompt_templates.py`  
**Responsibility:** Centralized prompt management (see Section 13)

---

### Module 5: `audio/` — Text-to-Speech Output

**File:** `audio/tts_engine.py`  
**Responsibility:** Convert translated text to audio

Key classes:
- `TTSEngine` — Factory that returns gTTS or pyttsx3 backend
- `gTTSBackend` — Online TTS using Google
- `pyttsx3Backend` — Offline TTS fallback

Key functions:
- `synthesize(text, lang_code)` → Returns audio bytes (MP3)
- `get_lang_code(language_name)` → Maps display name to BCP-47 code

Language → gTTS code mapping:
```
Hindi    → "hi"
Tamil    → "ta"
Bengali  → "bn"
Telugu   → "te"
Marathi  → "mr"
English  → "en"
```

---

### Module 6: `app/` — Streamlit Application

**File:** `app/main.py`  
**Responsibility:** Main Streamlit entry point, UI layout, state management

**File:** `app/components/` — Reusable UI components  
- `sidebar.py` — Language selector, settings, mode toggle
- `video_panel.py` — Webcam feed display with landmark overlay
- `output_panel.py` — Text and audio output display
- `history_panel.py` — Session history of recognized sentences

---

### Module 7: `training/` — Model Training Scripts

**File:** `training/data_collector.py` — Collect landmark data per sign label  
**File:** `training/preprocess.py` — Clean, augment, split dataset  
**File:** `training/train_static.py` — Train MLP/RF static classifier  
**File:** `training/train_dynamic.py` — Train LSTM dynamic classifier  
**File:** `training/evaluate.py` — Generate confusion matrix, accuracy reports  

---

### Module 8: `config/` — Configuration Management

**File:** `config/settings.py`  
```python
OPENAI_API_KEY: str  (from .env)
OPENAI_MODEL: str = "gpt-4.1-mini"
CONFIDENCE_THRESHOLD: float = 0.80
FRAME_SEQUENCE_LENGTH: int = 30
SENTENCE_PAUSE_TIMEOUT: float = 1.5  # seconds
SUPPORTED_LANGUAGES: list = ["Hindi", "Tamil", "Bengali", "Telugu", "Marathi"]
TTS_BACKEND: str = "gtts"  # or "pyttsx3"
```

---

## 7. AI/ML Pipeline Design

### 7.1 MediaPipe Hand Landmark Feature Vector

Each hand provides 21 3D landmarks. For 2 hands: 42 landmarks × 3 coords = **126 raw features**.

Normalization steps:
1. Set wrist (landmark 0) as origin → subtract wrist coords from all landmarks
2. Scale by hand bounding box diagonal → makes scale-invariant
3. Flatten to 1D array: `(126,)` for both hands, `(63,)` for single hand

Optional: Append z-score normalized velocity vectors across last N frames.

### 7.2 Static Classifier Architecture

```
Input: (63,) normalized landmark vector
       ↓
[Random Forest: 200 trees, max_depth=15]
[MLP: Dense(256) → ReLU → Dropout(0.3) → Dense(128) → ReLU → Dense(N_classes) → Softmax]
       ↓
Output: class probabilities (N_classes,)
```

Ensemble: Average probabilities of RF + MLP. Final label = argmax.

### 7.3 Dynamic Classifier Architecture (LSTM)

```
Input: (30, 63) — sequence of 30 frames, each with 63 features
       ↓
[Bidirectional LSTM: units=128, return_sequences=True]
       ↓
[Dropout: 0.3]
       ↓
[Bidirectional LSTM: units=64, return_sequences=False]
       ↓
[Dense: 64, ReLU]
       ↓
[Dropout: 0.2]
       ↓
[Dense: N_classes, Softmax]
       ↓
Output: class probabilities (N_classes,)
```

Training config:
- Loss: Categorical crossentropy
- Optimizer: Adam (lr=1e-3)
- Epochs: 100 (early stopping patience=15)
- Augmentation: Time warping, random noise injection, horizontal flip

### 7.4 Confidence Gating Logic

```python
def gate_prediction(static_conf, dynamic_conf, motion_detected):
    if motion_detected:
        primary = dynamic_conf
        fallback = static_conf
    else:
        primary = static_conf
        fallback = dynamic_conf

    final_conf = 0.7 * primary + 0.3 * fallback
    is_confident = final_conf >= CONFIDENCE_THRESHOLD
    return final_conf, is_confident
```

---

## 8. GPT-4.1 Mini Integration

### Why GPT-4.1 Mini?

ISL grammar differs significantly from English grammar. ISL uses a Topic-Comment structure (e.g., "APPLE I EAT" instead of "I eat the apple"). A classifier alone would produce grammatically incorrect output. GPT-4.1 Mini bridges this gap by:

1. **Grammar correction:** Converting ISL gloss → natural English
2. **Translation:** English → 5 Indian languages (zero-shot, high quality)
3. **Context awareness:** Understanding ambiguous signs from context
4. **Cost efficiency:** GPT-4.1 Mini is significantly cheaper than GPT-4o while maintaining high NLP quality

### Integration Architecture

```python
# nlp/gpt_client.py
from openai import OpenAI

client = OpenAI(api_key=OPENAI_API_KEY)

def call_gpt(system_prompt: str, user_message: str, max_tokens: int = 300) -> str:
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        max_tokens=max_tokens,
        temperature=0.3,  # Low temperature for deterministic translation
    )
    return response.choices[0].message.content.strip()
```

### Two-Stage GPT Pipeline

**Stage 1 — ISL Gloss → English:**

```
System: "You are an Indian Sign Language (ISL) interpreter. 
         ISL follows Topic-Comment grammar, different from English.
         Convert ISL gloss word sequences into natural, grammatically 
         correct English sentences. Preserve the original meaning exactly."

User: "ISL signs sequence: TOMORROW MARKET GO I VEGETABLE BUY"

Expected Output: "I will go to the market tomorrow to buy vegetables."
```

**Stage 2 — English → Target Language:**

```
System: "You are a professional translator. Translate the following 
         English sentence to {language}. Output ONLY the translation, 
         no explanation, no romanization, no extra text."

User: "I will go to the market tomorrow to buy vegetables."

Expected Output (Hindi): "मैं कल सब्ज़ियाँ खरीदने बाज़ार जाऊँगा।"
```

### Rate Limiting & Caching

- Cache translations: `dict {(gloss_tuple, lang): translated_text}` per session
- Retry with exponential backoff on API errors
- Fallback: Return English text if translation fails

---

## 9. Multilingual Support Plan

### Supported Languages (MVP v1.0)

| # | Language | Script | gTTS Code | Region |
|---|----------|--------|-----------|--------|
| 1 | Hindi | Devanagari | `hi` | North India |
| 2 | Tamil | Tamil | `ta` | Tamil Nadu, Sri Lanka |
| 3 | Bengali | Bengali | `bn` | West Bengal, Bangladesh |
| 4 | Telugu | Telugu | `te` | Andhra Pradesh, Telangana |
| 5 | Marathi | Devanagari | `mr` | Maharashtra |

### Language Selection Flow

1. User selects target language from sidebar dropdown
2. Language code stored in `st.session_state["target_language"]`
3. All subsequent translations sent to GPT-4.1 Mini with target language
4. gTTS synthesizes audio in the selected language's voice

### Future Languages (v2.0)

Kannada (`kn`), Malayalam (`ml`), Gujarati (`gu`), Punjabi (`pa`), Urdu (`ur`), English (`en`)

---

## 10. Streamlit App Design

### App Layout

```
┌────────────────────────────────────────────────────────────┐
│  🤟 ISL Multilingual Bridge                    [Settings]  │
├──────────────────┬─────────────────────────────────────────┤
│   SIDEBAR        │           MAIN PANEL                    │
│                  │  ┌──────────────────────────────────┐   │
│ 🌐 Language      │  │      LIVE CAMERA FEED            │   │
│ [ Hindi ▼ ]      │  │   (with landmark overlay)        │   │
│                  │  │   Hand skeleton drawn in green    │   │
│ 🎙 TTS Voice     │  └──────────────────────────────────┘   │
│ [ gTTS   ▼ ]     │                                         │
│                  │  ┌──────────────────────────────────┐   │
│ ⚙️ Settings      │  │  📝 Recognized Signs Buffer      │   │
│ Confidence: 0.80 │  │  [ HELLO ] [ MY ] [ NAME ] [ _ ] │   │
│ Pause: 1.5s      │  └──────────────────────────────────┘   │
│                  │                                         │
│ 📊 Stats         │  ┌──────────────────────────────────┐   │
│ Signs: 47        │  │  🇬🇧 English Output              │   │
│ Sentences: 3     │  │  "Hello, my name is Priya."      │   │
│                  │  └──────────────────────────────────┘   │
│ [▶ START]        │                                         │
│ [⏹ STOP ]        │  ┌──────────────────────────────────┐   │
│ [🔄 RESET]       │  │  🇮🇳 Hindi Translation           │   │
│                  │  │  "नमस्ते, मेरा नाम प्रिया है।"  │   │
│                  │  └──────────────────────────────────┘   │
│                  │                                         │
│                  │  ┌──────────────────────────────────┐   │
│                  │  │  🔊 Audio Output                 │   │
│                  │  │  [▶ Play] [⬇ Download]           │   │
│                  │  └──────────────────────────────────┘   │
│                  │                                         │
│                  │  📜 Session History                     │
│                  │  ───────────────────────────────────    │
│                  │  1. "Hello, my name is Priya."          │
│                  │     [नमस्ते, मेरा नाम प्रिया है।]     │
│                  │  2. "I need water please."              │
│                  │     [मुझे पानी चाहिए।]                 │
└──────────────────┴─────────────────────────────────────────┘
```

### Streamlit State Management

```python
# Session state variables
st.session_state["running"] = False           # Camera active flag
st.session_state["sign_buffer"] = []          # Current sentence signs
st.session_state["english_output"] = ""       # Latest English sentence
st.session_state["translated_output"] = ""    # Latest translation
st.session_state["audio_bytes"] = None        # Latest audio
st.session_state["history"] = []              # List of past sentences
st.session_state["target_language"] = "Hindi" # Selected language
st.session_state["confidence_threshold"] = 0.80
```

### Key Streamlit Components Used

- `streamlit-webrtc` — Real-time webcam access in browser
- `st.camera_input()` — Alternative for single-frame capture mode
- `st.audio()` — Audio playback
- `st.columns()` — Side-by-side layout
- `st.sidebar` — Settings panel
- `st.session_state` — Persistent app state across reruns
- `st.spinner()` — Loading indicator during API calls
- `st.metric()` — Stats display (signs recognized, confidence score)

---

## 11. Directory Structure

```
isl-multilingual-bridge/
│
├── app/
│   ├── main.py                     # Streamlit entry point
│   └── components/
│       ├── sidebar.py
│       ├── video_panel.py
│       ├── output_panel.py
│       └── history_panel.py
│
├── capture/
│   └── webcam_stream.py
│
├── vision/
│   └── landmark_extractor.py
│
├── recognition/
│   ├── static_classifier.py
│   ├── dynamic_classifier.py
│   ├── ensemble.py
│   └── sign_buffer.py
│
├── nlp/
│   ├── gpt_client.py
│   ├── sentence_builder.py
│   ├── translator.py
│   └── prompt_templates.py
│
├── audio/
│   └── tts_engine.py
│
├── training/
│   ├── data_collector.py
│   ├── preprocess.py
│   ├── train_static.py
│   ├── train_dynamic.py
│   └── evaluate.py
│
├── models/
│   ├── static_classifier.pkl       # Trained static sign classifier
│   ├── dynamic_classifier.h5       # Trained LSTM model
│   └── label_encoder.pkl           # Class label mapping
│
├── data/
│   ├── raw/                        # Raw collected landmark CSVs
│   ├── processed/                  # Cleaned, split datasets
│   └── isl_sign_list.json          # Full list of supported ISL signs
│
├── config/
│   └── settings.py
│
├── tests/
│   ├── test_vision.py
│   ├── test_recognition.py
│   ├── test_nlp.py
│   └── test_audio.py
│
├── docs/
│   ├── implementation_plan.md      # This document
│   └── api_reference.md
│
├── .env                            # API keys (gitignored)
├── .env.example                    # Template for .env
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## 12. Phase-wise Implementation Roadmap

### Phase 0 — Environment Setup (Week 1)

- [ ] Initialize GitHub repository with branch protection
- [ ] Set up Python virtual environment (Python 3.10+)
- [ ] Install core dependencies: mediapipe, opencv-python, tensorflow, streamlit, openai, gtts
- [ ] Configure `.env` with OpenAI API key
- [ ] Set up project folder structure as described in Section 11
- [ ] Create `.env.example` and `requirements.txt`

**Deliverables:** Working development environment, base project scaffold

---

### Phase 1 — Vision Pipeline (Week 2-3)

- [ ] Implement `webcam_stream.py`: OpenCV capture, frame preprocessing
- [ ] Implement `landmark_extractor.py`: MediaPipe Holistic integration
- [ ] Test hand landmark extraction on self-video
- [ ] Implement normalization: wrist-relative coordinates, scale normalization
- [ ] Add landmark overlay visualization (draw skeleton on frame)
- [ ] Integrate `streamlit-webrtc` for browser-based webcam access
- [ ] Verify 30fps processing without frame drops on target hardware

**Deliverables:** Streamlit page showing live webcam with hand skeleton overlay

---

### Phase 2 — Data Collection & Training (Week 3-5)

- [ ] Design ISL sign list (MVP: 100 signs — alphabets, common words, phrases)
- [ ] Build `data_collector.py`: Record N landmarks per sign, save as CSV
- [ ] Collect minimum 200 samples per sign (1 signer initially)
- [ ] Implement data augmentation: noise, scaling, mirroring
- [ ] Train and evaluate static classifier (target: >85% accuracy)
- [ ] Collect dynamic sign sequences (30-frame windows per dynamic word)
- [ ] Train LSTM dynamic classifier (target: >80% accuracy)
- [ ] Save trained models to `models/` directory

**Deliverables:** Trained models for 100 ISL signs, evaluation report

---

### Phase 3 — Recognition Engine (Week 5-6)

- [ ] Implement `static_classifier.py`: Load model, run inference
- [ ] Implement `dynamic_classifier.py`: Frame buffer, sequence inference
- [ ] Implement `ensemble.py`: Combine predictions, apply confidence gate
- [ ] Implement `sign_buffer.py`: Accumulate signs, deduplication, sentence trigger
- [ ] Unit test: Run recognition on pre-recorded ISL videos
- [ ] Tune confidence threshold (ROC analysis)
- [ ] Add motion detection heuristic for static/dynamic routing

**Deliverables:** End-to-end sign recognition module with buffering

---

### Phase 4 — GPT-4.1 Mini Integration (Week 6-7)

- [ ] Set up OpenAI client in `gpt_client.py` with retry logic
- [ ] Write and test prompt templates in `prompt_templates.py`
- [ ] Implement `sentence_builder.py`: ISL gloss → English
- [ ] Test sentence builder with 20+ diverse ISL gloss sequences
- [ ] Implement `translator.py`: English → 5 target languages
- [ ] Test translation quality for all 5 languages
- [ ] Implement session-level translation cache
- [ ] Add error handling: API timeout, rate limit, empty response

**Deliverables:** Working NLP pipeline, sample translations for all 5 languages

---

### Phase 5 — Audio Synthesis (Week 7)

- [ ] Implement `tts_engine.py` with gTTS backend
- [ ] Map language names to gTTS language codes
- [ ] Implement pyttsx3 offline fallback
- [ ] Test audio quality for all 5 languages
- [ ] Return audio as bytes for Streamlit `st.audio()` display
- [ ] Add download button for audio output

**Deliverables:** Audio output working for all 5 languages

---

### Phase 6 — Streamlit App Integration (Week 8)

- [ ] Build `app/main.py`: Layout, state management, orchestration
- [ ] Build sidebar: Language selector, TTS settings, confidence controls
- [ ] Integrate webcam stream with landmark overlay in app
- [ ] Display sign buffer as live chips/tags
- [ ] Display English and translated text output
- [ ] Integrate audio player with auto-play option
- [ ] Add session history panel with copy/export
- [ ] Add loading spinners for GPT and TTS calls
- [ ] Implement START / STOP / RESET controls
- [ ] Apply custom CSS for professional appearance
- [ ] Mobile responsiveness check

**Deliverables:** Fully functional Streamlit application

---

### Phase 7 — Testing & Refinement (Week 9)

- [ ] End-to-end integration testing with 5 different signers
- [ ] Measure: recognition accuracy, translation quality, latency
- [ ] User acceptance testing with 3 deaf community volunteers
- [ ] Fix identified issues, retrain if needed
- [ ] Performance profiling: identify bottlenecks
- [ ] Optimize frame processing pipeline

**Deliverables:** Tested, refined application with benchmark report

---

### Phase 8 — Deployment (Week 10)

- [ ] Write `Dockerfile` and `docker-compose.yml`
- [ ] Deploy to Streamlit Cloud or Hugging Face Spaces
- [ ] Configure secrets management for OpenAI key
- [ ] Write `README.md` with setup and usage instructions
- [ ] Record demo video

**Deliverables:** Publicly accessible application, demo video

---

## 13. API & Prompt Engineering

### Complete Prompt Templates

```python
# nlp/prompt_templates.py

ISL_TO_ENGLISH_SYSTEM = """
You are an expert Indian Sign Language (ISL) interpreter with deep knowledge 
of ISL grammar and linguistics. 

Key ISL grammar rules:
- ISL uses Topic-Comment word order (object before verb)
- Time expressions come first ("TOMORROW I GO" = "I will go tomorrow")
- Pronouns often omitted when context is clear
- No articles (a, an, the) in ISL
- Questions marked by facial expression, not word order

Your task: Convert the raw ISL sign sequence (gloss notation) into a 
natural, grammatically correct English sentence. Preserve exact meaning.
Output ONLY the English sentence, nothing else.
"""

ISL_TO_ENGLISH_USER = "ISL gloss sequence: {gloss}"

TRANSLATION_SYSTEM = """
You are a professional translator specializing in Indian languages.
Translate the given English text to {language} naturally and accurately.
Preserve the tone and intent of the original.
Output ONLY the {language} translation. No romanization, no explanation.
"""

TRANSLATION_USER = "Translate to {language}: {english_text}"

CONTEXT_ENRICHMENT_SYSTEM = """
You are interpreting Indian Sign Language in real-time. Given a sequence of 
recognized signs that may contain errors or missing context, produce the most 
likely intended meaning as a natural English sentence.
Consider common ISL conversational patterns. If uncertain, produce the most
plausible interpretation.
Output ONLY the interpreted English sentence.
"""
```

### GPT Call Orchestration

```python
# nlp/sentence_builder.py

async def build_sentence(sign_buffer: list[str]) -> dict:
    gloss = " ".join(sign_buffer).upper()
    
    # Stage 1: ISL → English
    english = await call_gpt_async(
        system=ISL_TO_ENGLISH_SYSTEM,
        user=ISL_TO_ENGLISH_USER.format(gloss=gloss),
        max_tokens=150
    )
    
    # Stage 2: English → Target Language
    target_lang = st.session_state["target_language"]
    translated = await call_gpt_async(
        system=TRANSLATION_SYSTEM.format(language=target_lang),
        user=TRANSLATION_USER.format(language=target_lang, english_text=english),
        max_tokens=200
    )
    
    return {
        "gloss": gloss,
        "english": english,
        "translated": translated,
        "language": target_lang
    }
```

---

## 14. Dataset Strategy

### ISL Sign List (MVP — 100 Signs)

**Category 1: ISL Alphabet (26 signs)**  
A through Z — static hand shapes

**Category 2: Numbers (10 signs)**  
0-9

**Category 3: Common Greetings & Phrases (15 signs)**  
HELLO, GOODBYE, THANK YOU, PLEASE, SORRY, YES, NO, HELP, STOP, WAIT, AGAIN, MORE, GOOD, BAD, OK

**Category 4: Basic Needs (15 signs)**  
WATER, FOOD, EAT, DRINK, SLEEP, SICK, PAIN, DOCTOR, TOILET, HOME, SCHOOL, WORK, MONEY, TIME, PHONE

**Category 5: People & Pronouns (10 signs)**  
I/ME, YOU, HE/SHE, WE, THEY, MOTHER, FATHER, FRIEND, CHILD, TEACHER

**Category 6: Actions (15 signs)**  
GO, COME, WANT, LIKE, LOVE, KNOW, UNDERSTAND, SEE, HEAR, SPEAK, READ, WRITE, GIVE, TAKE, BUY

**Category 7: Descriptors (9 signs)**  
BIG, SMALL, HOT, COLD, FAST, SLOW, HAPPY, SAD, ANGRY

### Data Collection Protocol

- Minimum: 200 samples per sign
- Capture under 3 lighting conditions (bright, dim, mixed)
- 5+ different signers (to improve generalization)
- Both left-hand and right-hand dominant signers
- File format: CSV with columns `[frame_id, lm_0_x, lm_0_y, lm_0_z, ..., lm_20_z, label]`

### External Datasets to Explore

- INCLUDE Dataset (IIT Madras) — ISL word-level dataset
- ISL-CSLRT Dataset — Continuous signing
- OpenPose-based ISL datasets on Kaggle

---

## 15. Testing & Evaluation Plan

### Unit Tests

| Module | Test | Tool |
|--------|------|------|
| `landmark_extractor.py` | Verify output shape for sample image | pytest |
| `static_classifier.py` | Load model, predict 5 sample vectors | pytest |
| `dynamic_classifier.py` | Predict from 30-frame sequence | pytest |
| `sentence_builder.py` | Mock GPT call, verify output format | pytest + mock |
| `translator.py` | Test all 5 language outputs | pytest |
| `tts_engine.py` | Verify non-empty audio bytes returned | pytest |

### Integration Tests

- Full pipeline test: image → landmarks → sign → sentence → translation → audio
- Streamlit app smoke test: confirm all UI components render
- Language switching: verify output changes correctly on language select

### Model Evaluation

- Confusion matrix across all 100 sign classes
- Per-class precision, recall, F1
- Real-time FPS benchmark (target: >20fps on standard laptop)
- Latency breakdown: vision (ms) + model inference (ms) + GPT API (ms) + TTS (ms)

### User Study

- 5 deaf/hard-of-hearing participants
- 5 hearing participants (non-signers as listeners)
- Tasks: demonstrate 10 common sentences in ISL, verify listener understood
- Measure: task completion rate, translation accuracy (subjective), system satisfaction (SUS questionnaire)

---

## 16. Deployment Plan

### Option A: Streamlit Cloud (Recommended for MVP)

```bash
# Requirements
- GitHub repo (public or private with Streamlit Cloud access)
- Streamlit Cloud account
- Add OPENAI_API_KEY in Streamlit Cloud Secrets Manager

# Steps
1. Push code to GitHub
2. Login to share.streamlit.io
3. New app → connect GitHub repo → set main file: app/main.py
4. Add secret: OPENAI_API_KEY = "sk-..."
5. Deploy
```

### Option B: Hugging Face Spaces

```bash
# Use Gradio or Streamlit Space
# Upload models to HF Hub for large model files
# Set OPENAI_API_KEY as Space secret
```

### Option C: Docker + VPS

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501
CMD ["streamlit", "run", "app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

```bash
docker build -t isl-bridge .
docker run -p 8501:8501 -e OPENAI_API_KEY=sk-... isl-bridge
```

### Camera Access Note

Streamlit Cloud and browser-based deployments require HTTPS for webcam access (browser security policy). Streamlit Cloud provides HTTPS by default.

---

## 17. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| MediaPipe fails to detect hands in poor lighting | High | High | Add lighting check, UI warning |
| GPT-4.1 Mini API downtime | Low | Medium | Retry logic, offline fallback (no translation) |
| Low recognition accuracy for new signers | Medium | High | Collect diverse training data, add fine-tuning option |
| gTTS audio quality poor for some languages | Medium | Medium | Add pyttsx3 fallback, allow user to select TTS engine |
| Webcam access blocked in browser | Low | High | Add fallback: upload video file for processing |
| ISL signs not in training set | High | Medium | Confidence gate rejects unknown signs, buffer shows "?" |
| Translation errors for regional dialects | Medium | Low | Prompt GPT to use standard formal dialect |
| API cost overrun (GPT-4.1 Mini) | Low | Medium | Cache translations per session, rate limit requests |

---

## 18. Future Roadmap

### v1.1 — Enhanced Recognition
- Expand sign vocabulary to 500+ words
- Multi-signer adaptation (personalized fine-tuning)
- Add facial expression recognition for ISL grammatical markers

### v1.2 — Reverse Direction
- Text → ISL avatar animation (ISL output for hearing → deaf communication)
- Sign language video generation using AI avatars

### v2.0 — Mobile App
- React Native / Flutter mobile app
- On-device ML inference (TFLite)
- Offline mode (no API dependency)

### v2.1 — Additional Language Support
- Expand to 10 Indian languages
- Add English audio output (for international use)
- Regional dialect support

### v3.0 — Continuous Signing
- Move from isolated word recognition to continuous sentence recognition
- CTC (Connectionist Temporal Classification) decoder for uninterrupted signing
- Real-time streaming without pause-based sentence triggers

---

## 📦 Requirements File

```
# Core Vision & ML
opencv-python>=4.8.0
mediapipe>=0.10.0
tensorflow>=2.13.0
scikit-learn>=1.3.0
numpy>=1.24.0
scipy>=1.11.0

# Streamlit App
streamlit>=1.28.0
streamlit-webrtc>=0.47.0
aiortc>=1.6.0

# AI / LLM
openai>=1.12.0

# Text-to-Speech
gTTS>=2.4.0
pyttsx3>=2.90

# Utilities
python-dotenv>=1.0.0
pillow>=10.0.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0

# Testing
pytest>=7.4.0
pytest-asyncio>=0.21.0
```

---

## 🚀 Quick Start Commands

```bash
# 1. Clone repo
git clone https://github.com/your-org/isl-multilingual-bridge.git
cd isl-multilingual-bridge

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env → add your OPENAI_API_KEY

# 5. Collect training data (run per sign label)
python training/data_collector.py --label HELLO --samples 200

# 6. Train models
python training/train_static.py
python training/train_dynamic.py

# 7. Launch Streamlit app
streamlit run app/main.py
```

---

*Document Version: 1.0.0 | Created: April 2026 | Project: ISL-Multilingual Bridge*
