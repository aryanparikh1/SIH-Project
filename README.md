# SIH26142 — Sentinel-2 Satellite Image Super-Resolution

> **Smart India Hackathon 2026** · Project ID: SIH26142

A web application that upscales Sentinel-2 satellite imagery from **10 m/px** to less than **4 m/px** using deep-learning super-resolution.

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the app

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

## Project Structure

```
sih_ui/
├── app.py                  # Home page — hero + upload
├── pages/
│   └── 1_Results.py        # Results page — enhanced image + comparison
├── utils/
│   ├── __init__.py
│   ├── model.py            # ⭐ ML model placeholder — edit this file
│   └── styles.py           # Custom CSS theme
├── .streamlit/
│   └── config.toml         # Streamlit theme configuration
├── requirements.txt
└── README.md
```

## Integrating Your ML Model

Open `utils/model.py` and replace the body of the `enhance_image()` function with your model inference code:

```python
def enhance_image(image: Image.Image) -> Image.Image:
    # Your model code here
    tensor = preprocess(image)
    with torch.no_grad():
        output = model(tensor)
    return postprocess(output)
```

That's the **only file** you need to modify. The rest of the app works automatically.

## Tech Stack

- **Streamlit** — Python web framework
- **Pillow** — Image processing
- **streamlit-image-comparison** — Interactive before/after slider
