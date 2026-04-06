# 🤟 ISL-Multilingual Bridge

> **Real-time Indian Sign Language → Multilingual Text & Audio Output**

An AI-powered pipeline that captures ISL gestures via webcam, recognizes signs using ML classifiers, converts to grammatical English using GPT-4.1 Mini, translates to 5 Indian languages, and synthesizes audio — all in a beautiful Streamlit web app.

---

## ✨ Features

- 📹 **Real-time webcam** sign language recognition via MediaPipe
- 🤖 **Dual ML classifiers** — Random Forest/MLP for static signs, LSTM for dynamic signs
- 🧠 **GPT-4.1 Mini** integration for ISL grammar correction & translation
- 🌐 **5 Indian languages** — Hindi, Tamil, Bengali, Telugu, Marathi
- 🔊 **Text-to-Speech** audio output with gTTS + offline fallback
- 🎨 **Premium dark UI** with glassmorphism design
- 📜 **Session history** with JSON export
- 🐳 **Docker ready** for easy deployment

---

## 🚀 Quick Start

### 1. Clone & Setup

```bash
git clone https://github.com/your-org/isl-multilingual-bridge.git
cd isl-multilingual-bridge

# Create virtual environment
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure

```bash
copy .env.example .env
# Edit .env → add your OPENAI_API_KEY
```

### 3. Launch App

```bash
streamlit run app/main.py
```

The app opens at `http://localhost:8501` with **Demo Mode** enabled (no trained models needed to test the UI + NLP pipeline).

---

## 📊 Train Models

### Collect Training Data

```bash
# Collect 200 samples of the "HELLO" sign
python training/data_collector.py --label HELLO --samples 200

# Collect dynamic sign sequences
python training/data_collector.py --label GO --samples 50 --mode dynamic
```

### Preprocess & Train

```bash
# Preprocess: augment + split data
python training/preprocess.py

# Train static classifier (Random Forest + MLP)
python training/train_static.py

# Train dynamic classifier (LSTM)
python training/train_dynamic.py

# Evaluate models
python training/evaluate.py
```

---

## 🏗 Architecture

```
User (ISL gesture) → Webcam → MediaPipe → Feature Extraction
   → Static/Dynamic Classifier → Sign Buffer
   → GPT-4.1 Mini (grammar + translation)
   → gTTS (audio synthesis) → Streamlit UI
```

---

## 📁 Project Structure

```
├── app/                    # Streamlit application
│   ├── main.py             # Entry point
│   └── components/         # UI components
├── capture/                # Webcam streaming
├── vision/                 # MediaPipe landmark extraction
├── recognition/            # Sign classification (static + dynamic + ensemble)
├── nlp/                    # GPT-4.1 Mini integration
├── audio/                  # Text-to-Speech
├── training/               # Data collection & model training
├── config/                 # Settings & configuration
├── models/                 # Trained model files
├── data/                   # Datasets & sign list
├── tests/                  # Unit tests
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## 🌐 Supported Languages

| Language | Code | Script |
|----------|------|--------|
| Hindi | `hi` | Devanagari |
| Tamil | `ta` | Tamil |
| Bengali | `bn` | Bengali |
| Telugu | `te` | Telugu |
| Marathi | `mr` | Devanagari |

---

## 🧪 Run Tests

```bash
pytest tests/ -v
```

---

## 🐳 Docker Deployment

```bash
docker-compose up --build
```

---

## 📝 License

MIT License

---

*Made with ❤️ for the deaf and hard-of-hearing community*
