"""
ISL-Multilingual Bridge — Sidebar Component
Language selector, TTS settings, confidence controls, and session stats.
"""

import streamlit as st
from config.settings import SUPPORTED_LANGUAGES, CONFIDENCE_THRESHOLD


def render_sidebar():
    """
    Render the sidebar with all controls and settings.
    Updates st.session_state with user selections.
    """
    with st.sidebar:
        # ── App Header ──────────────────────────────────
        st.markdown(
            """
            <div style="text-align: center; padding: 1rem 0;">
                <h1 style="font-size: 2rem; margin: 0;">🤟</h1>
                <h3 style="margin: 0.5rem 0; color: #E0E0E0;">ISL Bridge</h3>
                <p style="font-size: 0.75rem; color: #888;">
                    Indian Sign Language → Multilingual
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.divider()

        # ── Initialize defaults (before widgets) ─────────
        if "target_language" not in st.session_state:
            st.session_state["target_language"] = "Hindi"
        if "tts_backend" not in st.session_state:
            st.session_state["tts_backend"] = "gtts"
        if "auto_play" not in st.session_state:
            st.session_state["auto_play"] = True
        if "confidence_threshold" not in st.session_state:
            st.session_state["confidence_threshold"] = CONFIDENCE_THRESHOLD
        if "pause_timeout" not in st.session_state:
            st.session_state["pause_timeout"] = 1.5

        # ── Language Selector ────────────────────────────
        st.markdown("### 🌐 Target Language")
        languages = ["English"] + SUPPORTED_LANGUAGES
        st.selectbox(
            "Translate to:",
            languages,
            index=languages.index(st.session_state["target_language"]),
            key="target_language",
            label_visibility="collapsed",
        )

        st.divider()

        # ── TTS Settings ────────────────────────────────
        st.markdown("### 🎙 Voice Settings")
        tts_backend = st.radio(
            "TTS Engine:",
            ["gTTS (Online)", "pyttsx3 (Offline)"],
            index=0 if st.session_state["tts_backend"] == "gtts" else 1,
            key="tts_selector",
        )
        # Derive tts_backend from the radio widget value (this key is different)
        st.session_state["tts_backend"] = "gtts" if "gTTS" in tts_backend else "pyttsx3"

        st.checkbox("Auto-play audio", key="auto_play")

        st.divider()

        # ── Recognition Settings ─────────────────────────
        st.markdown("### ⚙️ Recognition")
        st.slider(
            "Confidence Threshold",
            min_value=0.5,
            max_value=0.99,
            step=0.05,
            key="confidence_threshold",
        )

        st.slider(
            "Sentence Pause (sec)",
            min_value=0.5,
            max_value=5.0,
            step=0.5,
            key="pause_timeout",
        )

        st.divider()

        # ── Session Stats ────────────────────────────────
        st.markdown("### 📊 Session Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "Signs",
                st.session_state.get("total_signs", 0),
            )
        with col2:
            st.metric(
                "Sentences",
                len(st.session_state.get("history", [])),
            )

        # ── Controls ─────────────────────────────────────
        st.divider()
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("▶", key="start_btn", use_container_width=True, help="Start camera"):
                st.session_state["running"] = True
        with col2:
            if st.button("⏹", key="stop_btn", use_container_width=True, help="Stop camera"):
                st.session_state["running"] = False
        with col3:
            if st.button("🔄", key="reset_btn", use_container_width=True, help="Reset session"):
                _reset_session()

        # ── Status indicator ─────────────────────────────
        is_running = st.session_state.get("running", False)
        status_color = "🟢" if is_running else "🔴"
        status_text = "Active" if is_running else "Stopped"
        st.markdown(f"**Status:** {status_color} {status_text}")


def _reset_session():
    """Reset the entire session state."""
    keys_to_reset = [
        "sign_buffer", "english_output", "translated_output",
        "audio_bytes", "history", "total_signs", "running"
    ]
    for key in keys_to_reset:
        if key in st.session_state:
            if key == "sign_buffer":
                st.session_state[key] = []
            elif key == "history":
                st.session_state[key] = []
            elif key == "running":
                st.session_state[key] = False
            elif key == "total_signs":
                st.session_state[key] = 0
            else:
                st.session_state[key] = ""
    st.toast("🔄 Session reset!", icon="✅")
