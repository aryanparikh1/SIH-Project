import streamlit as st
from utils.page_config import setup_page, render_footer

setup_page("References")

st.markdown("""
<div class="reveal">
  <p class="section-heading">References & Resources</p>
  <p class="section-sub">Literature, datasets, and tooling powering this project.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("### Research Papers")
st.markdown("""
<div class="ref-item">Dong, C., Loy, C. C., He, K., & Tang, X. (2015). Image super-resolution using deep convolutional networks. IEEE Transactions on Pattern Analysis and Machine Intelligence, 38(2), 295-307.</div>
<div class="ref-item">Lim, B., Son, S., Kim, H., Nah, S., & Mu Lee, K. (2017). Enhanced deep residual networks for single image super-resolution. In Proceedings of the IEEE conference on computer vision and pattern recognition workshops (pp. 136-144).</div>
<div class="ref-item">Wang, X., Yu, K., Wu, S., Gu, J., Liu, Y., Dong, C., ... & Change Loy, C. (2018). Esrgan: Enhanced super-resolution generative adversarial networks. In Proceedings of the European conference on computer vision (ECCV) workshops.</div>
<div class="ref-item">Li, J., Lu, Y., Liu, D., & Wang, Z. (2021). SwinIR: Image restoration using swin transformer. In Proceedings of the IEEE/CVF international conference on computer vision (pp. 1833-1844).</div>
<div class="ref-item">Zhang, Z., Liu, Q., & Wang, Y. (2018). Road extraction by deep residual u-net. IEEE Geoscience and Remote Sensing Letters, 15(5), 749-753.</div>
""", unsafe_allow_html=True)

st.markdown("### Datasets")
st.markdown("""
<div class="ref-item">
    <strong>Copernicus Sentinel-2 Data</strong><br>
    Open access high-resolution multispectral imagery provided by the European Space Agency (ESA). Utilized for the 10m/px source input data.
</div>
<div class="ref-item">
    <strong>WorldStrat Dataset</strong><br>
    A high-resolution, multi-sensor dataset for Earth Observation mapping and super-resolution. Provides the paired low/high-resolution tiles necessary for training the deep learning models.
</div>
""", unsafe_allow_html=True)

st.markdown("### Libraries & Frameworks")
st.markdown("""
<div class="ref-item">
    <strong>PyTorch & Torchvision</strong><br>
    Core deep learning framework used for designing, training, and running inference on all super-resolution architectures.
</div>
<div class="ref-item">
    <strong>Streamlit</strong><br>
    Python framework used to build this interactive web application and user interface.
</div>
<div class="ref-item">
    <strong>Rasterio</strong><br>
    Geospatial library utilized for reading and writing multi-band TIFF satellite imagery and preserving geographical metadata.
</div>
<div class="ref-item">
    <strong>Re (Regular Expressions)</strong><br>
    Python standard library module used for parsing metadata, filenames, and structuring unstructured satellite data logs.
</div>
""", unsafe_allow_html=True)

st.markdown("### Tools & Platforms")
st.markdown("""
<div class="ref-item">
    <strong>Kaggle</strong><br>
    Used for accessing computational resources (GPUs) required for training the deep learning models and hosting datasets.
</div>
<div class="ref-item">
    <strong>GitHub</strong><br>
    Version control, collaboration, and continuous integration platform for the project's codebase.
</div>
""", unsafe_allow_html=True)

render_footer()
