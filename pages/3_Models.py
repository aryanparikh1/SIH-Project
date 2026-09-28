import streamlit as st
from utils.page_config import setup_page, render_footer

setup_page("Models")

st.markdown("""
<div class="reveal">
  <p class="section-heading">Model Architectures</p>
  <p class="section-sub">Detailed overview of the super-resolution algorithms implemented in this project.</p>
</div>
""", unsafe_allow_html=True)

models_data = [
    {
        "name": "Bicubic Interpolation",
        "working": "A standard image scaling algorithm that computes the color of a new pixel by taking a weighted average of the 16 nearest pixels (a 4x4 neighborhood). It produces smoother edges than nearest-neighbor or bilinear interpolation.",
        "purpose": "Serves as the foundational baseline to compare deep learning models against. It is fast and requires no training, but struggles to reconstruct high-frequency details like building edges or crop boundaries.",
        "citation": "Keys, R. (1981). Cubic convolution interpolation for digital image processing. IEEE Transactions on Acoustics, Speech, and Signal Processing, 29(6), 1153-1160."
    },
    {
        "name": "SRCNN (Super-Resolution Convolutional Neural Network)",
        "working": "The first deep learning method for super-resolution. It uses a lightweight 3-layer CNN: patch extraction and representation, non-linear mapping, and reconstruction. It takes a low-resolution image upscaled to the target size via bicubic interpolation as input.",
        "purpose": "Used as the primary deep learning baseline. It demonstrates the fundamental advantage of learning mapping functions over mathematical interpolation, though it is limited by its shallow depth.",
        "citation": "Dong, C., Loy, C. C., He, K., & Tang, X. (2015). Image super-resolution using deep convolutional networks. IEEE Transactions on Pattern Analysis and Machine Intelligence, 38(2), 295-307."
    },
    {
        "name": "EDSR (Enhanced Deep Residual Networks)",
        "working": "EDSR improves upon standard ResNets by removing batch normalization modules from residual blocks. This prevents the network from losing color flexibility and saves memory, allowing for a much deeper network with more feature maps.",
        "purpose": "Provides high-fidelity reconstruction by leveraging deep residual learning. It is particularly effective at recovering textures in Sentinel-2 bands without introducing the artifacts sometimes seen in GANs.",
        "citation": "Lim, B., Son, S., Kim, H., Nah, S., & Mu Lee, K. (2017). Enhanced deep residual networks for single image super-resolution. In Proceedings of the IEEE conference on computer vision and pattern recognition workshops (pp. 136-144)."
    },
    {
        "name": "RUNet (Residual U-Net)",
        "working": "Combines the encoder-decoder architecture of U-Net with residual connections. The skip connections pass high-frequency spatial information from early layers to later layers, while residual blocks facilitate gradient flow through the deep network.",
        "purpose": "Specifically adapted for remote sensing imagery, RUNet is excellent at maintaining spatial coherence and edge sharpness, which is critical for urban planning and agricultural field boundary delineation.",
        "citation": "Zhang, Z., Liu, Q., & Wang, Y. (2018). Road extraction by deep residual u-net. IEEE Geoscience and Remote Sensing Letters, 15(5), 749-753."
    },
    {
        "name": "ESRGAN (Enhanced Super-Resolution Generative Adversarial Networks)",
        "working": "A GAN-based approach utilizing a generator built with Residual-in-Residual Dense Blocks (RRDB) without batch normalization. It employs a Relativistic average GAN discriminator and a perceptual loss function based on feature maps before activation.",
        "purpose": "Produces highly photorealistic results. While PSNR metrics might be slightly lower than EDSR, the perceptual quality is vastly superior, making it ideal for visual inspection of disaster zones or human-in-the-loop urban analysis.",
        "citation": "Wang, X., Yu, K., Wu, S., Gu, J., Liu, Y., Dong, C., ... & Change Loy, C. (2018). Esrgan: Enhanced super-resolution generative adversarial networks. In Proceedings of the European conference on computer vision (ECCV) workshops."
    },
    {
        "name": "SwinIR (Image Restoration Using Swin Transformer)",
        "working": "A state-of-the-art model that uses Swin Transformers (Shifted Window Transformers) instead of CNNs. It divides the image into non-overlapping local windows and computes self-attention within them, shifting the windows across layers to enable cross-window connections.",
        "purpose": "Serves as the flagship model for the project. By capturing long-range dependencies and global context, SwinIR excels at reconstructing complex, repetitive patterns found in satellite imagery (e.g., city grids, agricultural rows).",
        "citation": "Li, J., Lu, Y., Liu, D., & Wang, Z. (2021). SwinIR: Image restoration using swin transformer. In Proceedings of the IEEE/CVF international conference on computer vision (pp. 1833-1844)."
    }
]

import os

# Map model names to their architecture image files
ARCH_IMAGES = {
    "Bicubic Interpolation": "bicubic arch.jpeg",
    "SRCNN (Super-Resolution Convolutional Neural Network)": "srcnn-arch.jpeg",
    "EDSR (Enhanced Deep Residual Networks)": "edsr-arch.png",
    "RUNet (Residual U-Net)": "runet-arch.png",
    "ESRGAN (Enhanced Super-Resolution Generative Adversarial Networks)": "esrgan-arch.jpeg",
    "SwinIR (Image Restoration Using Swin Transformer)": "swinir-arch.png",
}

for model in models_data:
    with st.expander(model["name"]):
        img_file = ARCH_IMAGES.get(model["name"])
        if img_file:
            img_path = os.path.join("assets", img_file)
            if os.path.exists(img_path):
                st.image(img_path, caption=f'{model["name"]} Architecture', use_container_width=True)
            else:
                st.markdown(f'<div class="diagram-placeholder">Architecture image not found: {img_file}</div>', unsafe_allow_html=True)
        
        st.markdown("### How it Works")
        st.write(model["working"])
        
        st.markdown("### Purpose in Project")
        st.write(model["purpose"])
        
        st.markdown("### Citation")
        st.markdown(f'<div class="ref-item">{model["citation"]}</div>', unsafe_allow_html=True)

render_footer()
