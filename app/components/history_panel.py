"""
ISL-Multilingual Bridge — History Panel Component
Displays the session history of recognized and translated sentences.
"""

import streamlit as st
import json
from datetime import datetime


def render_history_panel():
    """
    Render the session history panel showing all past
    recognized sentences with their translations.
    """
    st.markdown(
        """
        <div style="
            background: linear-gradient(135deg, rgba(40,40,50,0.8), rgba(30,30,40,0.9));
            border: 1px solid rgba(150,150,200,0.2);
            border-radius: 12px;
            padding: 1rem;
            margin-top: 1rem;
        ">
            <h4 style="margin: 0 0 0.5rem 0; color: #B0B0D0;">📜 Session History</h4>
        </div>
        """,
        unsafe_allow_html=True,
    )

    history = st.session_state.get("history", [])

    if not history:
        st.markdown(
            """
            <p style="color: #555; font-style: italic; text-align: center; padding: 2rem 0;">
                No sentences yet. Start signing to see history here.
            </p>
            """,
            unsafe_allow_html=True,
        )
        return

    # Display history entries (newest first)
    for i, entry in enumerate(reversed(history)):
        idx = len(history) - i
        gloss = entry.get("gloss", "")
        english = entry.get("english", "")
        translated = entry.get("translated", "")
        language = entry.get("language", "Hindi")
        timestamp = entry.get("timestamp", "")

        st.markdown(
            f"""
            <div style="
                background: rgba(255,255,255,0.03);
                border: 1px solid rgba(100,100,150,0.15);
                border-radius: 10px;
                padding: 0.8rem 1rem;
                margin: 0.5rem 0;
            ">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="
                        background: rgba(100,100,255,0.2);
                        color: #A0A0FF;
                        padding: 0.15rem 0.5rem;
                        border-radius: 12px;
                        font-size: 0.7rem;
                        font-weight: 600;
                    ">#{idx}</span>
                    <span style="color: #666; font-size: 0.7rem;">{timestamp}</span>
                </div>
                <p style="
                    margin: 0.4rem 0 0.2rem;
                    color: #80B0FF;
                    font-size: 0.8rem;
                ">ISL: <code style="color: #ccc;">{gloss}</code></p>
                <p style="
                    margin: 0.2rem 0;
                    color: #E0E0E0;
                    font-size: 1rem;
                ">🇬🇧 {english}</p>
                <p style="
                    margin: 0.2rem 0 0;
                    color: #D080FF;
                    font-size: 1rem;
                ">🇮🇳 {translated}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Export / Clear ────────────────────────────────────
    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        # Export as JSON
        if st.button("📋 Export History", key="export_history", use_container_width=True):
            export_data = json.dumps(history, indent=2, ensure_ascii=False)
            st.download_button(
                label="⬇ Download JSON",
                data=export_data,
                file_name=f"isl_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                key="download_history",
            )

    with col2:
        if st.button("🗑 Clear History", key="clear_history", use_container_width=True):
            st.session_state["history"] = []
            st.toast("🗑 History cleared!", icon="✅")
            st.rerun()
