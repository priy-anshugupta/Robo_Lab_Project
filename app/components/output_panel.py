"""
ISL-Multilingual Bridge — Output Panel Component
Displays recognized signs buffer, English text, translation, and audio player.
"""

import streamlit as st


def render_output_panel():
    """
    Render the output panel with:
    - Sign buffer display (live chips/tags)
    - English text output
    - Translated text output
    - Audio player with download
    """
    # ── Sign Buffer Display ──────────────────────────────
    st.markdown(
        """
        <div style="
            background: linear-gradient(135deg, rgba(40,40,70,0.8), rgba(30,30,55,0.9));
            border: 1px solid rgba(100,200,100,0.2);
            border-radius: 12px;
            padding: 1rem;
            margin-bottom: 1rem;
        ">
            <h4 style="margin: 0 0 0.5rem 0; color: #80FF80;">📝 Recognized Signs</h4>
        </div>
        """,
        unsafe_allow_html=True,
    )

    sign_buffer = st.session_state.get("sign_buffer", [])
    latest_pred = st.session_state.get("latest_prediction") or {}
    latest_label = str(latest_pred.get("label", "") or "").strip().upper()
    latest_conf = float(latest_pred.get("confidence", 0.0) or 0.0)
    latest_ok = bool(latest_pred.get("is_confident", False))

    if latest_label:
        status_color = "#80FF80" if latest_ok else "#FFB366"
        status_text = "confident" if latest_ok else "low confidence"
        st.markdown(
            f'''
            <div style="
                background: rgba(255,255,255,0.04);
                border: 1px solid rgba(120,160,255,0.25);
                border-radius: 8px;
                padding: 0.6rem 0.8rem;
                margin-bottom: 0.6rem;
                color: #DCE6FF;
                font-size: 0.9rem;
            ">
                Current: <strong style="color:{status_color};">{latest_label}</strong>
                <span style="color:#9FB3D9;">({latest_conf:.0%}, {status_text})</span>
            </div>
            ''',
            unsafe_allow_html=True,
        )

    if sign_buffer:
        # Render as colorful chips
        chips_html = ""
        for sign in sign_buffer:
            chips_html += f"""
            <span style="
                display: inline-block;
                background: linear-gradient(135deg, #4A90D9, #357ABD);
                color: white;
                padding: 0.3rem 0.8rem;
                border-radius: 20px;
                margin: 0.2rem;
                font-weight: 600;
                font-size: 0.85rem;
                letter-spacing: 0.5px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.3);
            ">{sign}</span>
            """
        st.markdown(chips_html, unsafe_allow_html=True)
    else:
        st.markdown(
            '<p style="color: #666; font-style: italic;">Waiting for signs...</p>',
            unsafe_allow_html=True,
        )

    # Manual submit button
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("📤 Submit", key="submit_sentence", use_container_width=True):
            st.session_state["manual_submit"] = True

    st.markdown("---")

    # ── English Output ───────────────────────────────────
    english_output = st.session_state.get("english_output", "")

    st.markdown(
        """
        <div style="
            background: linear-gradient(135deg, rgba(40,50,70,0.8), rgba(30,40,60,0.9));
            border: 1px solid rgba(100,150,255,0.3);
            border-radius: 12px;
            padding: 1rem;
            margin-bottom: 1rem;
        ">
            <h4 style="margin: 0 0 0.5rem 0; color: #80B0FF;">🇬🇧 English</h4>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if english_output:
        st.markdown(
            f"""
            <div style="
                background: rgba(255,255,255,0.05);
                border-left: 3px solid #4A90D9;
                padding: 0.8rem 1rem;
                border-radius: 0 8px 8px 0;
                font-size: 1.1rem;
                color: #E0E0E0;
            ">{english_output}</div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<p style="color: #555; font-style: italic;">English output will appear here...</p>',
            unsafe_allow_html=True,
        )

    # ── Translated Output ────────────────────────────────
    translated_output = st.session_state.get("translated_output", "")
    target_lang = st.session_state.get("target_language", "Hindi")

    # Language flag emoji mapping
    lang_flags = {
        "Hindi": "🇮🇳", "Tamil": "🇮🇳", "Bengali": "🇮🇳",
        "Telugu": "🇮🇳", "Marathi": "🇮🇳", "English": "🇬🇧",
    }
    flag = lang_flags.get(target_lang, "🌐")

    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, rgba(50,40,70,0.8), rgba(40,30,60,0.9));
            border: 1px solid rgba(200,100,255,0.3);
            border-radius: 12px;
            padding: 1rem;
            margin-bottom: 1rem;
        ">
            <h4 style="margin: 0 0 0.5rem 0; color: #D080FF;">
                {flag} {target_lang} Translation
            </h4>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if translated_output:
        st.markdown(
            f"""
            <div style="
                background: rgba(255,255,255,0.05);
                border-left: 3px solid #9B59B6;
                padding: 0.8rem 1rem;
                border-radius: 0 8px 8px 0;
                font-size: 1.2rem;
                color: #E0E0E0;
                line-height: 1.6;
            ">{translated_output}</div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<p style="color: #555; font-style: italic;">{target_lang} translation will appear here...</p>',
            unsafe_allow_html=True,
        )

    # ── Audio Output ─────────────────────────────────────
    audio_bytes = st.session_state.get("audio_bytes", None)

    st.markdown(
        """
        <div style="
            background: linear-gradient(135deg, rgba(50,50,40,0.8), rgba(40,40,30,0.9));
            border: 1px solid rgba(255,200,100,0.3);
            border-radius: 12px;
            padding: 1rem;
            margin-bottom: 1rem;
        ">
            <h4 style="margin: 0 0 0.5rem 0; color: #FFD080;">🔊 Audio Output</h4>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if audio_bytes:
        st.audio(audio_bytes, format="audio/mp3")

        st.download_button(
            label="⬇ Download Audio",
            data=audio_bytes,
            file_name=f"isl_translation_{target_lang.lower()}.mp3",
            mime="audio/mp3",
            key="download_audio",
        )
    else:
        st.markdown(
            '<p style="color: #555; font-style: italic;">Audio will play here after translation...</p>',
            unsafe_allow_html=True,
        )
