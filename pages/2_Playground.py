import io
import streamlit as st
from PIL import Image

from utils.page_config import setup_page, render_footer
from utils.model import enhance_image, compute_tta_uncertainty, MODEL_LOADERS, DEVICE

setup_page("Playground")

st.markdown("""
<div class="reveal">
  <p class="section-heading">Model Playground</p>
  <p class="section-sub">Select a super-resolution model and upload an image to test its performance.</p>
</div>
""", unsafe_allow_html=True)

# Model Selection
model_category = st.radio(
    "Select Model Category",
    ["Baseline & Early Deep Learning", "GAN-based", "Transformer-based"],
    horizontal=True,
    label_visibility="collapsed"
)

if "GAN" in model_category:
    selected_model = "ESRGAN"
elif "Transformer" in model_category:
    selected_model = "SwinIR"
else:
    selected_model = st.selectbox("Select Model", ["Bicubic Interpolation", "SRCNN", "RUNet", "EDSR"])


st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# Image Upload
uploaded = st.file_uploader(
    "Upload Sentinel-2 image (.tiff only, 10 m/px)",
    type=["tif", "tiff"],
)

if uploaded:
    try:
        import rasterio
        from rasterio.io import MemoryFile
        import numpy as np
        import torch
        
        raw_tensor = None  # Will hold raw reflectance [C, H, W] for GAN
        
        with MemoryFile(uploaded.getvalue()) as memfile:
            with memfile.open() as dataset:
                raw_data = dataset.read()
                num_bands = raw_data.shape[0]
                original_dtype = raw_data.dtype
                
                # --- Build raw_tensor for GAN (raw reflectance / 10000, [0,1]) ---
                # Model expects [B02, B03, B04, B08] order — keep bands as-is from TIFF
                if num_bands >= 4:
                    raw_float = raw_data[:4].astype(np.float32)
                    # Normalize (do NOT reorder — model trained on [B02,B03,B04,B08])
                    if np.issubdtype(original_dtype, np.integer) or raw_float.max() > 100:
                        raw_float = raw_float / 10000.0
                    raw_float = np.nan_to_num(raw_float, nan=0.0, posinf=1.0, neginf=0.0)
                    raw_float = np.clip(raw_float, 0.0, 1.0)
                    raw_tensor = torch.from_numpy(raw_float)  # [4, H, W]
                elif num_bands >= 3:
                    raw_float = raw_data[:3].astype(np.float32)
                    # No reordering — feed as-is
                    if np.issubdtype(original_dtype, np.integer) or raw_float.max() > 100:
                        raw_float = raw_float / 10000.0
                    raw_float = np.nan_to_num(raw_float, nan=0.0, posinf=1.0, neginf=0.0)
                    raw_float = np.clip(raw_float, 0.0, 1.0)
                    raw_tensor = torch.from_numpy(raw_float)  # [3, H, W]
                
                # --- Build display images (percentile-stretched 8-bit) ---
                if num_bands == 1:
                    rgb_data = np.repeat(raw_data, 3, axis=0)
                elif num_bands >= 3:
                    rgb_data = raw_data[:3, :, :]
                    # Reorder [B02, B03, B04] -> [B04, B03, B02]
                    rgb_data = rgb_data[[2, 1, 0]]
                else:
                    rgb_data = raw_data[:1, :, :]
                    rgb_data = np.repeat(rgb_data, 3, axis=0)
                    
                rgb_data = np.transpose(rgb_data, (1, 2, 0)).astype(np.float32)
                
                # Normalize RGB to 8-bit for display
                p2, p98 = np.percentile(rgb_data, (2, 98))
                if p98 > p2:
                    rgb_data = np.clip(rgb_data, p2, p98)
                    rgb_data = ((rgb_data - p2) / (p98 - p2) * 255.0).astype(np.uint8)
                else:
                    rgb_data = np.zeros_like(rgb_data, dtype=np.uint8)
                        
                image = Image.fromarray(rgb_data).convert("RGB")
                
                # Extract NIR band for display (4th band if available)
                image_nir = None
                if num_bands >= 4:
                    nir_band = raw_data[3, :, :].astype(np.float32)
                    p2n, p98n = np.percentile(nir_band, (2, 98))
                    if p98n > p2n:
                        nir_band = np.clip(nir_band, p2n, p98n)
                        nir_band = ((nir_band - p2n) / (p98n - p2n) * 255.0).astype(np.uint8)
                    else:
                        nir_band = np.zeros_like(nir_band, dtype=np.uint8)
                    image_nir = Image.fromarray(nir_band, mode="L")
                    
    except Exception as e:
        st.error(f"Error reading TIFF file: {e}")
        st.stop()
    
    is_gan = selected_model == "ESRGAN"
    
    if is_gan:
        # GAN path: return dict with separate RGB, NIR, uncertainty
        with st.spinner(f"Running {selected_model}..."):
            result = enhance_image(image, model_type=selected_model, image_nir=image_nir, return_dict=True, raw_tensor=raw_tensor)
        
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="reveal">
          <p class="section-heading">GAN Results</p>
        </div>
        """, unsafe_allow_html=True)
        
        # --- RGB Output ---
        st.markdown('<p style="font-size:1.1rem;font-weight:600;color:#e2e8f0;margin-bottom:0.5rem;">RGB Output</p>', unsafe_allow_html=True)
        
        rgb_orig = result["rgb"]["original"]
        rgb_enh = result["rgb"]["enhanced"]
        rgb_orig_upscaled = rgb_orig.resize((rgb_enh.width, rgb_enh.height), resample=Image.NEAREST)
        
        try:
            from streamlit_image_comparison import image_comparison
            image_comparison(
                img1=rgb_orig_upscaled,
                img2=rgb_enh,
                label1="Original RGB",
                label2="Enhanced RGB (GAN)",
                width=700,
                show_labels=True,
                make_responsive=True,
            )
        except ImportError:
            c1, c2 = st.columns(2)
            with c1:
                st.image(rgb_orig_upscaled, caption="Original RGB", use_container_width=True)
            with c2:
                st.image(rgb_enh, caption="Enhanced RGB (GAN)", use_container_width=True)
        
        st.markdown('<br>', unsafe_allow_html=True)
        
        # --- NIR Output ---
        st.markdown('<p style="font-size:1.1rem;font-weight:600;color:#e2e8f0;margin-bottom:0.5rem;">NIR Output</p>', unsafe_allow_html=True)
        
        if result["nir"] is not None:
            nir_orig = result["nir"]["original"]
            nir_enh = result["nir"]["enhanced"]
            
            # Convert to RGB for the slider (grayscale displayed as 3-channel)
            nir_orig_rgb = nir_orig.convert("RGB")
            nir_enh_rgb = nir_enh.convert("RGB")
            nir_orig_upscaled = nir_orig_rgb.resize((nir_enh_rgb.width, nir_enh_rgb.height), resample=Image.NEAREST)
            
            try:
                from streamlit_image_comparison import image_comparison
                image_comparison(
                    img1=nir_orig_upscaled,
                    img2=nir_enh_rgb,
                    label1="Original NIR",
                    label2="Enhanced NIR (GAN)",
                    width=700,
                    show_labels=True,
                    make_responsive=True,
                )
            except ImportError:
                c1, c2 = st.columns(2)
                with c1:
                    st.image(nir_orig_upscaled, caption="Original NIR", use_container_width=True)
                with c2:
                    st.image(nir_enh_rgb, caption="Enhanced NIR (GAN)", use_container_width=True)
        else:
            st.info("NIR band not available in the uploaded TIFF. Upload a 4+ band Sentinel-2 image to see NIR output.")
        
        st.markdown('<br>', unsafe_allow_html=True)

        # --- Uncertainty Mapping ---
        st.markdown('<p style="font-size:1.1rem;font-weight:600;color:#e2e8f0;margin-bottom:0.5rem;">Uncertainty Mapping</p>', unsafe_allow_html=True)
        _, u_col, _ = st.columns([1, 2, 1])
        with u_col:
            if result["uncertainty"] is not None:
                st.image(result["uncertainty"], caption="GAN Uncertainty Map", use_container_width=True)
            else:
                st.info("Uncertainty map not yet generated by the current GAN model. This section will display model confidence data once the uncertainty head is integrated.")

        st.markdown('<br>', unsafe_allow_html=True)

        # --- Error Metrics ---
        st.markdown('<p style="font-size:1.1rem;font-weight:600;color:#e2e8f0;margin-bottom:0.5rem;">Error Metrics (Placeholder)</p>', unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown('<div class="stat-card"><div class="stat-value">28.45</div><div class="stat-label">PSNR (dB)</div></div>', unsafe_allow_html=True)
        with m2:
            st.markdown('<div class="stat-card"><div class="stat-value">0.862</div><div class="stat-label">SSIM</div></div>', unsafe_allow_html=True)
        with m3:
            st.markdown('<div class="stat-card"><div class="stat-value">0.012</div><div class="stat-label">MSE</div></div>', unsafe_allow_html=True)
    
    else:
        # Non-GAN path: TTA uncertainty for SRCNN, RUNet, EDSR, SwinIR
        import torch
        import torch.nn.functional as F
        from torchvision import transforms as T

        with st.spinner(f"Running {selected_model} with TTA uncertainty (6 augmentations)..."):
            # Load model
            loader = MODEL_LOADERS.get(selected_model)
            model, device = loader() if loader else (None, None)

            if model is not None and raw_tensor is not None:
                if selected_model == "SwinIR":
                    # SwinIR uses percentile-stretch normalization (not /10000)
                    # and was trained on [B02, B03, B04] order (BGR).
                    input_np = raw_tensor[:3].numpy()
                    input_np = np.nan_to_num(input_np, nan=0.0, posinf=0.0, neginf=0.0)
                    p2 = np.percentile(input_np, 2)
                    p98 = np.percentile(input_np, 98)
                    if p98 > p2:
                        input_np = (input_np - p2) / (p98 - p2)
                    elif np.max(input_np) > 0:
                        input_np = input_np / np.max(input_np)
                    input_np = np.clip(input_np, 0.0, 1.0)
                    input_t = torch.from_numpy(input_np).float().unsqueeze(0).to(device)
                else:
                    # Baseline models: reorder [B02,B03,B04] → [B04,B03,B02]
                    input_t = raw_tensor[:3].clone()
                    input_t = input_t[[2, 1, 0]]
                    input_t = input_t.unsqueeze(0).to(device)

                bicubic_input = selected_model in ["SRCNN", "RUNet"]
                mean_sr, unc_img, conf_img = compute_tta_uncertainty(
                    model, input_t, bicubic_input=bicubic_input, scale=4
                )

                # Convert mean SR to display image
                sr_out = mean_sr[0].cpu().numpy().transpose(1, 2, 0)  # [H, W, 3]

                if selected_model == "SwinIR":
                    # SwinIR processed native [B02, B03, B04] order.
                    # Convert output BGR back to RGB [B04, B03, B02] for display.
                    sr_out = sr_out[:, :, [2, 1, 0]]
                    
                # Global percentile stretch to maintain visual consistency with input
                p2 = np.percentile(sr_out, 2)
                p98 = np.percentile(sr_out, 98)
                if p98 > p2:
                    sr_out = (sr_out - p2) / (p98 - p2 + 1e-8)

                sr_out = np.clip(sr_out, 0, 1)
                enhanced = Image.fromarray((sr_out * 255).astype(np.uint8))
            elif model is not None:
                # Fallback: PIL image input (no raw_tensor)
                enhanced = enhance_image(image, model_type=selected_model)
                unc_img = None
                conf_img = None
            else:
                enhanced = image
                unc_img = None
                conf_img = None

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="reveal">
          <p class="section-heading">Results</p>
        </div>
        """, unsafe_allow_html=True)

        # 1. Comparison slider
        st.markdown('<p style="font-size:1.1rem;font-weight:600;color:#e2e8f0;margin-bottom:0.5rem;">Image Comparison</p>', unsafe_allow_html=True)
        
        original_upscaled = image.resize((enhanced.width, enhanced.height), resample=Image.NEAREST)
        
        try:
            from streamlit_image_comparison import image_comparison
            image_comparison(
                img1=original_upscaled,
                img2=enhanced,
                label1="Original Input",
                label2=f"Enhanced ({selected_model})",
                width=700,
                show_labels=True,
                make_responsive=True,
            )
        except ImportError:
            c1, c2 = st.columns(2)
            with c1:
                st.image(original_upscaled, caption="Original Input", use_container_width=True)
            with c2:
                st.image(enhanced, caption=f"Enhanced ({selected_model})", use_container_width=True)

        st.markdown('<br>', unsafe_allow_html=True)

        # 2. Uncertainty & Confidence Maps
        if unc_img is not None and conf_img is not None:
            st.markdown('<p style="font-size:1.1rem;font-weight:600;color:#e2e8f0;margin-bottom:0.5rem;">Uncertainty & Confidence Maps</p>', unsafe_allow_html=True)
            u_col, c_col = st.columns(2)
            with u_col:
                st.image(unc_img, caption=f"{selected_model} Uncertainty (hot colormap)", use_container_width=True)
            with c_col:
                st.image(conf_img, caption=f"{selected_model} Confidence (viridis colormap)", use_container_width=True)
        else:
            st.markdown('<p style="font-size:1.1rem;font-weight:600;color:#e2e8f0;margin-bottom:0.5rem;">Uncertainty Mapping</p>', unsafe_allow_html=True)
            st.info("Uncertainty mapping requires a TIFF file with raw reflectance data.")

        st.markdown('<br>', unsafe_allow_html=True)

        # 3. Error Metrics
        st.markdown('<p style="font-size:1.1rem;font-weight:600;color:#e2e8f0;margin-bottom:0.5rem;">Error Metrics (Placeholder)</p>', unsafe_allow_html=True)
        
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown('<div class="stat-card"><div class="stat-value">28.45</div><div class="stat-label">PSNR (dB)</div></div>', unsafe_allow_html=True)
        with m2:
            st.markdown('<div class="stat-card"><div class="stat-value">0.862</div><div class="stat-label">SSIM</div></div>', unsafe_allow_html=True)
        with m3:
            st.markdown('<div class="stat-card"><div class="stat-value">0.012</div><div class="stat-label">MSE</div></div>', unsafe_allow_html=True)

render_footer()
