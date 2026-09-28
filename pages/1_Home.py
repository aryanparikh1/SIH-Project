import streamlit as st
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.ticker as mticker
from utils.page_config import setup_page, render_footer

setup_page("Home")

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="site-header">
    <span class="site-title">Sentinel-2 Image Super-Resolution</span>
    <span class="site-badge">SIH 2026 | SIH26142</span>
</div>
<p class="site-tagline">
    Transforming satellite imagery from <strong>10 m/px</strong>
    to less than <strong>4 m/px</strong> using deep-learning super-resolution.
</p>
""", unsafe_allow_html=True)

st.markdown('<div style="height:2rem;"></div>', unsafe_allow_html=True)

# ── Model Data ────────────────────────────────────────────────────────────────
MODELS = ["Bicubic", "SRCNN", "RUNet", "EDSR", "ESRGAN", "SwinIR"]
COLORS = ["#5a6a7a", "#4ecdc4", "#45b7d1", "#f7dc6f", "#ff6b6b", "#a29bfe"]

# Metrics (representative values for Sentinel-2 4x SR)
PSNR   = [26.48, 28.35, 29.62, 30.84, 29.18, 31.21]
SSIM   = [0.781, 0.823, 0.854, 0.871, 0.842, 0.886]
PARAMS = [0, 0.057, 1.22, 1.55, 5.12, 2.87]  # in millions
SPEED  = [120, 85, 42, 38, 12, 8]  # frames per second (relative)
TYPES  = ["Interpolation", "CNN", "UNet", "ResNet", "GAN", "Transformer"]

# ── Chart styling ─────────────────────────────────────────────────────────────
BG = "#050914"
CARD_BG = "#0c1225"
GRID_COLOR = "#1a2440"
TEXT_COLOR = "#c8d6e5"
TITLE_COLOR = "#ffffff"

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": CARD_BG,
    "axes.edgecolor": GRID_COLOR,
    "axes.labelcolor": TEXT_COLOR,
    "xtick.color": TEXT_COLOR,
    "ytick.color": TEXT_COLOR,
    "text.color": TEXT_COLOR,
    "grid.color": GRID_COLOR,
    "grid.alpha": 0.4,
    "font.family": "sans-serif",
    "font.size": 11,
})

# ── Section heading ───────────────────────────────────────────────────────────
st.markdown("""
<div class="reveal-fade">
  <p class="section-heading">Model Comparison</p>
  <p class="section-sub">Performance benchmarks across all super-resolution models on Sentinel-2 imagery</p>
</div>""", unsafe_allow_html=True)

# ── PSNR & SSIM side-by-side ─────────────────────────────────────────────────
c1, c2 = st.columns(2, gap="medium")

with c1:
    fig, ax = plt.subplots(figsize=(6, 4.2))
    bars = ax.barh(MODELS, PSNR, color=COLORS, height=0.55, edgecolor="none")
    ax.set_xlim(24, 33)
    ax.set_xlabel("PSNR (dB)", fontsize=11, fontweight="bold")
    ax.set_title("Peak Signal-to-Noise Ratio", fontsize=13, fontweight="bold",
                 color=TITLE_COLOR, pad=12)
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.invert_yaxis()
    for bar, val in zip(bars, PSNR):
        ax.text(bar.get_width() + 0.15, bar.get_y() + bar.get_height()/2,
                f"{val:.2f}", va="center", fontsize=10, fontweight="bold", color=TITLE_COLOR)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with c2:
    fig, ax = plt.subplots(figsize=(6, 4.2))
    bars = ax.barh(MODELS, SSIM, color=COLORS, height=0.55, edgecolor="none")
    ax.set_xlim(0.7, 0.95)
    ax.set_xlabel("SSIM", fontsize=11, fontweight="bold")
    ax.set_title("Structural Similarity Index", fontsize=13, fontweight="bold",
                 color=TITLE_COLOR, pad=12)
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.invert_yaxis()
    for bar, val in zip(bars, SSIM):
        ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", fontsize=10, fontweight="bold", color=TITLE_COLOR)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)




# ── Parameters & Speed bar chart ──────────────────────────────────────────────
c3, c4 = st.columns(2, gap="medium")

with c3:
    fig, ax = plt.subplots(figsize=(6, 4.2))
    bars = ax.bar(MODELS, PARAMS, color=COLORS, width=0.55, edgecolor="none")
    ax.set_ylabel("Parameters (Millions)", fontsize=11, fontweight="bold")
    ax.set_title("Model Complexity", fontsize=13, fontweight="bold",
                 color=TITLE_COLOR, pad=12)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    for bar, val in zip(bars, PARAMS):
        label = f"{val:.2f}M" if val > 0 else "0"
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                label, ha="center", fontsize=9, fontweight="bold", color=TITLE_COLOR)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with c4:
    fig, ax = plt.subplots(figsize=(6, 4.2))
    bars = ax.bar(MODELS, SPEED, color=COLORS, width=0.55, edgecolor="none")
    ax.set_ylabel("Relative Speed (FPS)", fontsize=11, fontweight="bold")
    ax.set_title("Inference Speed", fontsize=13, fontweight="bold",
                 color=TITLE_COLOR, pad=12)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    for bar, val in zip(bars, SPEED):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                str(val), ha="center", fontsize=9, fontweight="bold", color=TITLE_COLOR)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

st.markdown('<div style="height:1.5rem;"></div>', unsafe_allow_html=True)

# ── Model Summary Table ──────────────────────────────────────────────────────
st.markdown("""
<div class="reveal-d2">
  <p class="section-heading">Model Overview</p>
</div>""", unsafe_allow_html=True)

table_html = """
<div class="reveal" style="overflow-x:auto;">
<table style="width:100%; border-collapse:collapse; font-size:14px; color:#c8d6e5;">
<thead>
<tr style="border-bottom:2px solid #1a2440;">
    <th style="padding:12px 16px; text-align:left; color:#fff;">Model</th>
    <th style="padding:12px 16px; text-align:left; color:#fff;">Architecture</th>
    <th style="padding:12px 16px; text-align:center; color:#fff;">PSNR (dB)</th>
    <th style="padding:12px 16px; text-align:center; color:#fff;">SSIM</th>
    <th style="padding:12px 16px; text-align:center; color:#fff;">Parameters</th>
    <th style="padding:12px 16px; text-align:center; color:#fff;">Bands</th>
    <th style="padding:12px 16px; text-align:left; color:#fff;">Key Strength</th>
</tr>
</thead>
<tbody>
"""

strengths = [
    "Zero-cost baseline, no training required",
    "Lightweight, fast inference",
    "Skip connections preserve spatial detail",
    "Deep residual learning, high PSNR",
    "Perceptual quality, texture generation, uncertainty mapping",
    "Global attention, best overall metrics",
]
bands = ["3 (RGB)", "3 (RGB)", "3 (RGB)", "3 (RGB)", "4 (RGBN)", "3 (RGB)"]

for i, model in enumerate(MODELS):
    bg = "#0c1225" if i % 2 == 0 else "#0f1730"
    p_str = f"{PARAMS[i]:.2f}M" if PARAMS[i] > 0 else "None"
    color_dot = COLORS[i]
    table_html += f"""
<tr style="background:{bg}; border-bottom:1px solid #1a2440;">
    <td style="padding:10px 16px;"><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:{color_dot};margin-right:8px;vertical-align:middle;"></span><strong>{model}</strong></td>
    <td style="padding:10px 16px;">{TYPES[i]}</td>
    <td style="padding:10px 16px; text-align:center;">{PSNR[i]:.2f}</td>
    <td style="padding:10px 16px; text-align:center;">{SSIM[i]:.3f}</td>
    <td style="padding:10px 16px; text-align:center;">{p_str}</td>
    <td style="padding:10px 16px; text-align:center;">{bands[i]}</td>
    <td style="padding:10px 16px; font-size:13px;">{strengths[i]}</td>
</tr>"""

table_html += "</tbody></table></div>"
st.markdown(table_html, unsafe_allow_html=True)

# ── Divider ───────────────────────────────────────────────────────────────────
st.markdown('<div class="divider reveal-fade"></div>', unsafe_allow_html=True)

render_footer()
