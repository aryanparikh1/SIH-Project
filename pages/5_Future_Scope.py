import streamlit as st
from utils.page_config import setup_page, render_footer

setup_page("Future Scope")

st.markdown("""
<div class="reveal">
  <p class="section-heading">Future Scope & Novel Angles</p>
  <p class="section-sub">Roadmap for advancing the super-resolution pipeline.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("### 1. Advanced Generative Models")
st.write("""
While GANs and Transformers currently lead our pipeline, **Diffusion Models (e.g., SR3, StableSR)** represent the next frontier. 
Implementing latent diffusion models specifically trained on Earth Observation data could resolve hallucination issues present in GANs, offering more reliable textures for scientific analysis. 
Additionally, exploring **Neural Radiance Fields (NeRFs)** using multi-angle satellite passes could allow for 3D super-resolved topographical reconstruction.
""")

st.markdown("### 2. Multi-Temporal & Spectral Fusion")
st.write("""
Currently, the models perform Single Image Super-Resolution (SISR). A novel angle is implementing **Multi-Image Super-Resolution (MISR)** by exploiting sub-pixel shifts across a time-series of Sentinel-2 revisits (every 5 days). 
Furthermore, fusing the 10m bands (RGB, NIR) with the 20m/60m bands (SWIR, Coastal Aerosol) using **cross-attention mechanisms** could yield a fully super-resolved, hyperspectral data cube.
""")

st.markdown("### 3. Physics-Informed Super-Resolution")
st.write("""
Deep learning models often ignore the physical properties of the sensor. By incorporating the **Point Spread Function (PSF)** and Modulation Transfer Function (MTF) of the Sentinel-2 MultiSpectral Instrument (MSI) directly into the loss function, we can constrain the network to produce images that are physically consistent with actual optics, reducing artifacts.
""")

st.markdown("### 4. Edge Deployment & Real-Time Inference")
st.write("""
To make the solution highly accessible, future iterations will focus on **model quantization (INT8) and pruning**. 
Deploying lightweight models (like an optimized RUNet) directly to edge devices (e.g., drones or field tablets) or integrating them as WebAssembly (WASM) plugins for GIS platforms (QGIS, ArcGIS) would allow researchers to upscale imagery locally without relying on cloud infrastructure.
""")

render_footer()
