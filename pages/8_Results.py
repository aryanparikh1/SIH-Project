"""
SIH26142 — Sentinel-2 Satellite Image Super-Resolution
=======================================================
Page 8: Results — Enhanced image display, comparison slider, metadata, download
"""

import io

import streamlit as st
from PIL import Image

from utils.page_config import setup_page, render_footer

setup_page("Results")

# ── Guard: redirect if no images in session ────────────────────
if "enhanced_image" not in st.session_state or "original_image" not in st.session_state:
    st.markdown(
        """
        <div class="result-header">
            <div class="result-title">No Image Found</div>
            <div class="result-subtitle">
                Please upload and enhance an image from the Home page first.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("")
    _, btn_col, _ = st.columns([1, 1, 1])
    with btn_col:
        if st.button("Go to Home", type="primary", use_container_width=True):
            st.switch_page("pages/1_Home.py")
    st.stop()


# ── Retrieve images from session state ─────────────────────────
original: Image.Image = st.session_state["original_image"]
enhanced: Image.Image = st.session_state["enhanced_image"]
filename: str = st.session_state.get("original_filename", "image.png")


# ── Header ─────────────────────────────────────────────────────
st.markdown(
    """
    <div class="result-header">
        <div class="success-badge">Enhancement Complete</div>
        <div class="result-title">Your Enhanced Image</div>
        <div class="result-subtitle">
            Super-resolution has been applied successfully. Compare and download below.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ── Metadata Cards ─────────────────────────────────────────────
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns(4, gap="medium")

scale_w = enhanced.width / original.width if original.width else 0
scale_h = enhanced.height / original.height if original.height else 0
avg_scale = (scale_w + scale_h) / 2

with m1:
    st.markdown(
        f"""
        <div class="metadata-card">
            <div class="metadata-label">Original Size</div>
            <div class="metadata-value">{original.width} x {original.height} px</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with m2:
    st.markdown(
        f"""
        <div class="metadata-card">
            <div class="metadata-label">Enhanced Size</div>
            <div class="metadata-value-accent">{enhanced.width} x {enhanced.height} px</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with m3:
    st.markdown(
        f"""
        <div class="metadata-card">
            <div class="metadata-label">Scale Factor</div>
            <div class="metadata-value-accent">{avg_scale:.1f}x</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with m4:
    # Estimated resolution improvement
    orig_res = 10.0  # metres
    new_res = orig_res / avg_scale if avg_scale else orig_res
    st.markdown(
        f"""
        <div class="metadata-card">
            <div class="metadata-label">Est. Resolution</div>
            <div class="metadata-value-accent">{new_res:.1f} m/px</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Enhanced Image Display + Download ──────────────────────────
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

st.markdown(
    '<p style="text-align:center;font-size:1.4rem;font-weight:700;color:#e2e8f0;'
    'margin-bottom:0.8rem;">Enhanced Image</p>',
    unsafe_allow_html=True,
)

_, img_col, _ = st.columns([0.5, 3, 0.5])
with img_col:
    st.image(enhanced, caption="Enhanced Image", use_container_width=True)

# Download button
_, dl_col, _ = st.columns([1, 1, 1])
with dl_col:
    buf = io.BytesIO()
    enhanced.save(buf, format="PNG")
    enhanced_name = filename.rsplit(".", 1)[0] + "_enhanced.png"

    st.download_button(
        label="Download Enhanced Image",
        data=buf.getvalue(),
        file_name=enhanced_name,
        mime="image/png",
        type="primary",
        use_container_width=True,
    )


# ── Comparison Section ─────────────────────────────────────────
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

st.markdown(
    '<div class="compare-title">Before and After Comparison</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="compare-desc">Drag the slider to compare original vs enhanced image.</div>',
    unsafe_allow_html=True,
)

# Resize original to match enhanced dimensions for fair comparison
# Using NEAREST to keep it pixelated — shows the quality difference clearly
original_upscaled = original.resize(
    (enhanced.width, enhanced.height),
    resample=Image.NEAREST,
)

# Try to use the streamlit-image-comparison component
try:
    from streamlit_image_comparison import image_comparison

    _, comp_col, _ = st.columns([0.5, 3, 0.5])
    with comp_col:
        image_comparison(
            img1=original_upscaled,
            img2=enhanced,
            label1="Original (10 m/px)",
            label2="Enhanced (<4 m/px)",
            width=700,
            show_labels=True,
            make_responsive=True,
        )

except ImportError:
    # Fallback: side-by-side columns
    st.warning(
        "Install `streamlit-image-comparison` for the interactive slider: "
        "`pip install streamlit-image-comparison`",
    )
    left, right = st.columns(2)
    with left:
        st.image(
            original_upscaled,
            caption="Original (10 m/px)",
            use_container_width=True,
        )
    with right:
        st.image(enhanced, caption="Enhanced (<4 m/px)", use_container_width=True)


# ── Side-by-side static view ──────────────────────────────────
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

st.markdown(
    '<p style="text-align:center;font-size:1.2rem;font-weight:600;color:#94a3b8;'
    'margin-bottom:0.8rem;">Side-by-Side View</p>',
    unsafe_allow_html=True,
)

left_col, right_col = st.columns(2, gap="medium")
with left_col:
    st.image(original, caption="Original Image", use_container_width=True)
with right_col:
    st.image(enhanced, caption="Enhanced Image", use_container_width=True)


# ── Back Button ────────────────────────────────────────────────
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

_, back_col, _ = st.columns([1, 1, 1])
with back_col:
    if st.button("Enhance Another Image", use_container_width=True):
        # Clear session state
        for key in ["original_image", "enhanced_image", "original_filename"]:
            st.session_state.pop(key, None)
        st.switch_page("pages/1_Home.py")


render_footer()
