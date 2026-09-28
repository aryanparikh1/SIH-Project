import streamlit as st
import streamlit.components.v1 as components
from utils.styles import CUSTOM_CSS, SPACE_BG_HTML, THREEJS_INJECT

def setup_page(title: str):
    """
    Sets up the common page configuration for all pages.
    Includes page config, CSS, background, and the 3D globe injection.
    """
    # Page config is now handled in app.py

    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.markdown(SPACE_BG_HTML, unsafe_allow_html=True)

    # Inject High-Res Three.js Globe
    components.html(THREEJS_INJECT, height=0)

def render_footer():
    """Renders the common footer across all pages."""
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="reveal-fade">
      <p style="text-align:center;color:#334155;font-size:.78rem;padding:.8rem 0;">
        SIH26142 | Smart India Hackathon 2026 |
        Sentinel-2 Satellite Image Super-Resolution
      </p>
    </div>""", unsafe_allow_html=True)
