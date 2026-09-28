"""
ML Model Integration Point
===========================

This module contains the image enhancement function for the SIH26142 project.
Replace the placeholder implementation with your trained super-resolution model.

Current placeholder: Bicubic interpolation (2.5x upscale)
Target: ML super-resolution for Sentinel-2 imagery (10 m/px → < 4 m/px)
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
from utils.rrdbnet import RRDBNet
from utils.custom_transformer import SimpleTransformerSR
from utils.baseline_models import SRCNN, RUNet, EDSR
import streamlit as st
import os
import matplotlib
import matplotlib.colors

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── TTA Uncertainty Functions ─────────────────────────────────────────────────

def _apply_tta(x, index):
    """Apply one of 6 test-time augmentations."""
    if index == 0: return x
    elif index == 1: return torch.rot90(x, 1, dims=[2, 3])
    elif index == 2: return torch.rot90(x, 2, dims=[2, 3])
    elif index == 3: return torch.rot90(x, 3, dims=[2, 3])
    elif index == 4: return torch.flip(x, dims=[3])
    elif index == 5: return torch.flip(x, dims=[2])

def _inverse_tta(x, index):
    """Inverse of _apply_tta to bring prediction back to original orientation."""
    if index == 0: return x
    elif index == 1: return torch.rot90(x, 3, dims=[2, 3])
    elif index == 2: return torch.rot90(x, 2, dims=[2, 3])
    elif index == 3: return torch.rot90(x, 1, dims=[2, 3])
    elif index == 4: return torch.flip(x, dims=[3])
    elif index == 5: return torch.flip(x, dims=[2])

def compute_tta_uncertainty(model, input_tensor, bicubic_input=False, scale=4):
    """
    Run 6 TTA predictions, compute mean SR and uncertainty.
    Returns: (mean_sr [1,C,H,W], uncertainty_map [H,W], confidence_map [H,W])
    """
    predictions = []
    with torch.no_grad():
        for i in range(6):
            transformed = _apply_tta(input_tensor, i)
            if bicubic_input:
                _, _, h, w = transformed.shape
                transformed = F.interpolate(transformed, size=(h * scale, w * scale),
                                            mode="bicubic", align_corners=False)
            pred = model(transformed).clamp(0, 1)
            pred = _inverse_tta(pred, i)
            predictions.append(pred)

    preds = torch.stack(predictions, dim=0)  # [6, 1, C, H, W]
    mean_sr = preds.mean(dim=0)              # [1, C, H, W]

    # Pixel-wise std across TTA, averaged over RGB channels
    uncertainty_rgb = torch.std(preds, dim=0)  # [1, C, H, W]
    uncertainty = uncertainty_rgb.mean(dim=1)[0]  # [H, W]

    # Convert to numpy
    unc_np = uncertainty.detach().cpu().numpy()

    # Confidence = 1 - normalized_uncertainty
    p2 = np.percentile(unc_np, 2)
    p98 = np.percentile(unc_np, 98)
    norm_unc = np.clip((unc_np - p2) / (p98 - p2 + 1e-8), 0, 1)
    conf_np = 1.0 - norm_unc

    # Create uncertainty heatmap image (hot colormap)
    cmap_hot = matplotlib.colormaps["hot"]
    unc_color = (cmap_hot(norm_unc)[:, :, :3] * 255).astype(np.uint8)
    unc_img = Image.fromarray(unc_color)

    # Create confidence heatmap image (viridis colormap)
    cmap_viridis = matplotlib.colormaps["viridis"]
    conf_color = (cmap_viridis(conf_np)[:, :, :3] * 255).astype(np.uint8)
    conf_img = Image.fromarray(conf_color)

    return mean_sr, unc_img, conf_img


def _load_model(model_cls, model_path, cls_kwargs=None):
    """Generic loader: instantiate model_cls, load weights, return (model, device) or (None, None)."""
    model = model_cls(**(cls_kwargs or {}))
    if not os.path.exists(model_path):
        st.warning(f"Weights not found: {model_path}")
        return None, None
    try:
        sd = torch.load(model_path, map_location=DEVICE, weights_only=False)
        # Handle nested state dicts (ESRGAN style / basicSR / SwinIR)
        if isinstance(sd, dict):
            for key in ("params_ema", "params", "generator", "model", "state_dict"):
                if key in sd:
                    sd = sd[key]
                    break
        
        model.load_state_dict(sd, strict=True)
    except Exception as e:
        st.warning(f"Could not load weights from {model_path}: {e}")
        return None, None
    model.eval().to(DEVICE)
    return model, DEVICE


@st.cache_resource
def load_srcnn():
    return _load_model(SRCNN, "models/srcnn_best.pth")

@st.cache_resource
def load_runet():
    return _load_model(RUNet, "models/runet_best.pth")

@st.cache_resource
def load_edsr():
    return _load_model(EDSR, "models/edsr_best.pth")

@st.cache_resource
def load_esrgan():
    return _load_model(RRDBNet, "models/sentinel2_rrdbgan_epoch_29.pth",
                       cls_kwargs=dict(in_nc=4, out_nc=4, nf=64, nb=6, gc=32))

@st.cache_resource
def load_transformer():
    return _load_model(SimpleTransformerSR, "models/transformer_weights.pth",
                       cls_kwargs=dict(in_channels=3, out_channels=3, embed_dim=96,
                                       depth=6, heads=6, scale=4, dropout=0.10))


# Map UI names to loader functions
MODEL_LOADERS = {
    "SRCNN": load_srcnn,
    "RUNet": load_runet,
    "EDSR": load_edsr,
    "ESRGAN": load_esrgan,
    "SwinIR": load_transformer,
}


def _bicubic_upscale(image: Image.Image, scale: float = 4.0) -> Image.Image:
    new_w = int(image.width * scale)
    new_h = int(image.height * scale)
    return image.resize((new_w, new_h), resample=Image.BICUBIC)


def enhance_image(image: Image.Image, model_type: str = "Bicubic Interpolation", image_nir: Image.Image = None, return_dict: bool = False, raw_tensor: torch.Tensor = None):
    """
    raw_tensor: optional [C, H, W] float tensor with raw reflectance values in [0, 1].
                Used by the GAN model which was trained on raw Sentinel-2 data / 10000.
    """
    # Pure bicubic path (no neural network)
    if "Bicubic" in model_type:
        res = _bicubic_upscale(image.convert("RGB"))
        if return_dict:
            return {"rgb": {"original": image, "enhanced": res}, "nir": None, "uncertainty": None}
        return res

    # Resolve loader
    loader = MODEL_LOADERS.get(model_type)
    if loader is None:
        # Try partial match
        for name, fn in MODEL_LOADERS.items():
            if name in model_type or model_type in name:
                loader = fn
                break

    if loader is None:
        st.warning(f"Unknown model '{model_type}', falling back to Bicubic.")
        res = _bicubic_upscale(image.convert("RGB"))
        if return_dict:
            return {"rgb": {"original": image, "enhanced": res}, "nir": None, "uncertainty": None}
        return res

    model, device = loader()

    if model is None:
        res = _bicubic_upscale(image.convert("RGB"))
        if return_dict:
            return {"rgb": {"original": image, "enhanced": res}, "nir": None, "uncertainty": None}
        return res

    try:

        in_channels = 3
        out_channels = 3
        if hasattr(model, 'conv_first') and model.conv_first.in_channels == 4:
            in_channels = 4
            out_channels = 4

        # For 4-channel GAN, use raw_tensor if available
        if in_channels == 4 and raw_tensor is not None:
            img_tensor = raw_tensor.unsqueeze(0).to(device)
        else:
            img_rgb = image.convert("RGB")
            img_tensor = transforms.ToTensor()(img_rgb).unsqueeze(0).to(device)
            
            if model_type == "SwinIR":
                # SwinIR expects BGR order
                img_tensor = img_tensor[:, [2, 1, 0], :, :]
            
            if in_channels == 4:
                if image_nir is not None:
                    nir_tensor = transforms.ToTensor()(image_nir.convert("L")).unsqueeze(0).to(device)
                else:
                    nir_tensor = img_tensor[:, 0:1, :, :]
                img_tensor = torch.cat([img_tensor, nir_tensor], dim=1)

        # If it's SRCNN or RUNet, we must upscale first using bicubic
        if model_type in ["SRCNN", "RUNet"]:
            img_tensor = F.interpolate(img_tensor, scale_factor=4.0, mode="bicubic", align_corners=False)

        # Monte Carlo Uncertainty for GAN
        uncertainty_img = None
        if return_dict and "ESRGAN" in str(model_type):
            model.eval()
            has_dropout = False
            for module in model.modules():
                if isinstance(module, torch.nn.Dropout2d):
                    module.train()
                    has_dropout = True
            
            if has_dropout:
                predictions = []
                with torch.no_grad():
                    for _ in range(20):
                        sr = model(img_tensor).clamp(0, 1)
                        predictions.append(sr)
                predictions = torch.stack(predictions, dim=0)
                mean_sr = predictions.mean(dim=0)
                uncertainty = predictions.std(dim=0, unbiased=False)
                output = mean_sr[0]
                uncertainty = uncertainty[0]
            else:
                with torch.no_grad():
                    output = model(img_tensor).squeeze().clamp(0, 1)
                uncertainty = torch.zeros_like(output)
                
            # Create RGB uncertainty image with hot colormap
            rgb_uncertainty = uncertainty[:3].mean(dim=0)
            unc_min, unc_max = rgb_uncertainty.min(), rgb_uncertainty.max()
            if unc_max > unc_min:
                rgb_uncertainty = (rgb_uncertainty - unc_min) / (unc_max - unc_min)
            unc_array = (rgb_uncertainty.cpu().numpy() * 255).astype(np.uint8)
            cmap_hot = matplotlib.colormaps["hot"]
            unc_color = (cmap_hot(unc_array / 255.0)[:, :, :3] * 255).astype(np.uint8)
            uncertainty_img = Image.fromarray(unc_color)

        else:
            with torch.no_grad():
                output = model(img_tensor)
            output = output.squeeze().float().cpu().clamp_(0, 1)
        
        # Split output channels
        if out_channels == 4 and output.shape[0] == 4:
            out_rgb = output[:3, :, :]
            out_nir = output[3:4, :, :]
        else:
            out_rgb = output[:3, :, :]
            out_nir = None

        # For GAN output: model has sigmoid, values in [0,1] but small (~0.1-0.2).
        # Notebook uses sr_tensor[[0,1,2]] (no reordering) + per-channel percentile stretch.
        # PIL needs explicit stretch since it doesn't auto-scale like matplotlib imshow.
        if out_channels == 4:
            # Notebook sr_to_rgb: x[[0,1,2]] — SR output is already RGB, no reordering needed
            rgb_np = out_rgb[[0, 1, 2]].clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
            # Global percentile stretch for PIL display to preserve color balance
            p_low = np.percentile(rgb_np, 2)
            p_high = np.percentile(rgb_np, 98)
            if p_high > p_low:
                rgb_np = (rgb_np - p_low) / (p_high - p_low + 1e-8)
            rgb_np = np.clip(rgb_np, 0, 1)
            res_rgb = Image.fromarray((rgb_np * 255).astype(np.uint8))
            
            if out_nir is not None:
                nir_np = out_nir.squeeze().clamp(0, 1).cpu().numpy()
                p_low = np.percentile(nir_np, 2)
                p_high = np.percentile(nir_np, 98)
                nir_np = np.clip((nir_np - p_low) / (p_high - p_low + 1e-8), 0, 1)
                res_nir = Image.fromarray((nir_np * 255).astype(np.uint8), mode="L")
            else:
                res_nir = None
        else:
            sr_out = out_rgb.cpu().numpy().transpose(1, 2, 0)
            if model_type == "SwinIR":
                # Convert output BGR back to RGB
                sr_out = sr_out[:, :, [2, 1, 0]]
                
            # Global percentile stretch
            p2 = np.percentile(sr_out, 2)
            p98 = np.percentile(sr_out, 98)
            if p98 > p2:
                sr_out = (sr_out - p2) / (p98 - p2 + 1e-8)
            sr_out = np.clip(sr_out, 0, 1)
            res_rgb = Image.fromarray((sr_out * 255).astype(np.uint8))

            res_nir = transforms.ToPILImage()(out_nir) if out_nir is not None else None
        
        if return_dict:
            return {
                "rgb": {
                    "original": image,
                    "enhanced": res_rgb
                },
                "nir": {
                    "original": image_nir if image_nir is not None else image.convert("L"),
                    "enhanced": res_nir if res_nir is not None else res_rgb.convert("L")
                },
                "uncertainty": uncertainty_img
            }

        return res_rgb

    except Exception as e:
        st.error(f"Inference error ({model_type}): {e}")
        res = _bicubic_upscale(image.convert("RGB"))
        if return_dict:
            return {"rgb": {"original": image, "enhanced": res}, "nir": None, "uncertainty": None}
        return res
