import os
import sys
import numpy as np
import pandas as pd
from PIL import Image
import joblib
import streamlit as st

# Batas ukuran file maksimal (5MB)
MAX_FILE_SIZE_MB = 5

# Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title="Klasifikasi Kondisi Rak Sepatu",
    page_icon="👟",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom Styling: Abu-abu gelap matte & Glassmorphism bening dengan teks putih semi-transparan
st.markdown("""
<style>
    /* Latar Belakang Abu-abu Gelap Matte */
    .stApp {
        background-color: #202225 !important;
        background-image: radial-gradient(circle at 50% 15%, #2a2c31 0%, #1a1b1e 100%) !important;
        color: rgba(255, 255, 255, 0.85) !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
    }

    /* Sembunyikan elemen bawaan Streamlit */
    header[data-testid="stHeader"], footer, #MainMenu, .stDeployButton {
        display: none !important;
        visibility: hidden !important;
    }

    .block-container {
        max-width: 580px !important;
        padding-top: 3rem !important;
        padding-bottom: 3rem !important;
    }

    /* Container Kartu Utama Bening / Glassmorphic Bevel */
    .glass-card {
        background: rgba(36, 38, 42, 0.65);
        border: 1.5px solid rgba(255, 255, 255, 0.09);
        border-radius: 28px;
        padding: 32px 28px 24px 28px;
        backdrop-filter: blur(24px);
        -webkit-backdrop-filter: blur(24px);
        box-shadow: 0 25px 50px rgba(0, 0, 0, 0.6), inset 0 1px 1px rgba(255, 255, 255, 0.14);
        margin-bottom: 22px;
        text-align: center;
    }

    .title-text {
        color: rgba(255, 255, 255, 0.88);
        font-size: 24px;
        font-weight: 600;
        letter-spacing: -0.3px;
        margin-bottom: 6px;
    }

    .subtitle-text {
        color: rgba(255, 255, 255, 0.42);
        font-size: 13.5px;
        font-weight: 400;
        margin-bottom: 22px;
    }

    /* Styling Uploader Bening */
    [data-testid="stFileUploader"] {
        background: transparent !important;
    }
    [data-testid="stFileUploaderDropzone"] {
        background: rgba(26, 28, 31, 0.5) !important;
        border: 1.5px dashed rgba(255, 255, 255, 0.12) !important;
        border-radius: 22px !important;
        padding: 24px 16px !important;
        transition: all 0.25s ease !important;
    }
    [data-testid="stFileUploaderDropzone"]:hover {
        border-color: rgba(255, 255, 255, 0.28) !important;
        background: rgba(32, 34, 38, 0.6) !important;
    }
    [data-testid="stFileUploaderDropzoneInstructions"] {
        color: rgba(255, 255, 255, 0.42) !important;
    }
    [data-testid="stFileUploaderDropzone"] button {
        background: rgba(255, 255, 255, 0.08) !important;
        color: rgba(255, 255, 255, 0.82) !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 16px !important;
        font-size: 13px !important;
        padding: 6px 16px !important;
        backdrop-filter: blur(10px) !important;
    }
    [data-testid="stFileUploaderDropzone"] button:hover {
        background: rgba(255, 255, 255, 0.15) !important;
        border-color: rgba(255, 255, 255, 0.28) !important;
        color: rgba(255, 255, 255, 0.95) !important;
    }
    [data-testid="stFileUploaderFile"] {
        background: rgba(255, 255, 255, 0.04) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 14px !important;
        color: rgba(255, 255, 255, 0.7) !important;
    }

    /* Frame Gambar */
    [data-testid="stImage"] {
        display: flex !important;
        justify-content: center !important;
        margin-top: 15px !important;
    }
    [data-testid="stImage"] img {
        border-radius: 20px !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        box-shadow: 0 18px 36px rgba(0, 0, 0, 0.55) !important;
        max-height: 380px !important;
        object-fit: contain !important;
    }

    /* Kartu Hasil Klasifikasi Bening */
    .result-card {
        background: rgba(30, 32, 36, 0.7);
        border: 1.5px solid rgba(255, 255, 255, 0.1);
        border-radius: 24px;
        padding: 20px 24px;
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        box-shadow: 0 16px 36px rgba(0, 0, 0, 0.45), inset 0 1px 1px rgba(255, 255, 255, 0.12);
        margin-top: 20px;
        text-align: center;
    }
    .result-badge {
        display: inline-block;
        padding: 6px 20px;
        border-radius: 30px;
        font-size: 20px;
        font-weight: 700;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-top: 4px;
        margin-bottom: 4px;
    }
    .status-berantakan {
        background: rgba(235, 87, 87, 0.14);
        border: 1px solid rgba(235, 87, 87, 0.35);
        color: rgba(255, 140, 140, 0.95);
    }
    .status-rapi {
        background: rgba(39, 174, 96, 0.14);
        border: 1px solid rgba(39, 174, 96, 0.35);
        color: rgba(140, 255, 180, 0.95);
    }
    .status-kosong {
        background: rgba(45, 156, 219, 0.14);
        border: 1px solid rgba(45, 156, 219, 0.35);
        color: rgba(140, 215, 255, 0.95);
    }
    .result-subtext {
        color: rgba(255, 255, 255, 0.45);
        font-size: 13px;
        margin-top: 6px;
    }
</style>
""", unsafe_allow_html=True)

# Fungsi Ekstraksi GLCM (Sesuai proses ekstraksi fitur)
def extract_glcm_features(gray):
    levels = 256
    glcm = np.zeros(levels * levels, dtype=np.int32)
    glcm += np.bincount(((gray[:, :-1].astype(np.int32) << 8) | gray[:, 1:]).ravel(), minlength=65536)
    glcm += np.bincount(((gray[:-1, :].astype(np.int32) << 8) | gray[1:, :]).ravel(), minlength=65536)
    glcm += np.bincount(((gray[:-1, :-1].astype(np.int32) << 8) | gray[1:, 1:]).ravel(), minlength=65536)
    glcm += np.bincount(((gray[:-1, 1:].astype(np.int32) << 8) | gray[1:, :-1]).ravel(), minlength=65536)
    glcm = glcm.reshape((levels, levels)).astype(np.float64)
    glcm += glcm.T
    s = glcm.sum()
    if s > 0:
        glcm /= s

    i, j = np.indices((levels, levels), dtype=np.float64)
    contrast = float(np.sum(glcm * ((i - j) ** 2)))
    dissimilarity = float(np.sum(glcm * np.abs(i - j)))
    homogeneity = float(np.sum(glcm / (1.0 + (i - j) ** 2)))
    asm = float(np.sum(glcm ** 2))
    energy = float(np.sqrt(asm))

    mu_i = np.sum(i * glcm)
    mu_j = np.sum(j * glcm)
    var_i = np.sum(glcm * ((i - mu_i) ** 2))
    var_j = np.sum(glcm * ((j - mu_j) ** 2))
    std_prod = np.sqrt(var_i * var_j)
    correlation = float(np.sum(glcm * (i - mu_i) * (j - mu_j)) / std_prod if std_prod > 0 else 1.0)

    return {
        "glcm_contrast": contrast,
        "glcm_dissimilarity": dissimilarity,
        "glcm_homogeneity": homogeneity,
        "glcm_energy": energy,
        "glcm_correlation": correlation,
        "glcm_asm": asm,
    }

# Ekstraksi 16 Fitur Visual dari Gambar
def extract_features(pil_img):
    rgb = np.asarray(pil_img.convert("RGB"), dtype=np.float32)
    gray = np.asarray(pil_img.convert("L"), dtype=np.uint8)

    mean_rgb = rgb.mean(axis=(0, 1))
    std_rgb = rgb.std(axis=(0, 1), ddof=0)
    diff_rgb = rgb - mean_rgb
    std_safe = np.where(std_rgb == 0, 1e-10, std_rgb)
    skew_rgb = np.mean(diff_rgb ** 3, axis=(0, 1)) / (std_safe ** 3)

    diff_gray = gray.astype(np.float32) - gray.mean()
    std_gray = gray.std()
    skew_gray = float(np.mean(diff_gray ** 3) / (std_gray ** 3) if std_gray > 0 else 0.0)

    glcm_feats = extract_glcm_features(gray)

    row = {
        "R_mean": mean_rgb[0], "G_mean": mean_rgb[1], "B_mean": mean_rgb[2],
        "R_std": std_rgb[0], "G_std": std_rgb[1], "B_std": std_rgb[2],
        "R_skew": skew_rgb[0], "G_skew": skew_rgb[1], "B_skew": skew_rgb[2],
        "gray_skew": skew_gray,
        **glcm_feats
    }
    cols = [
        "R_mean", "G_mean", "B_mean", "R_std", "G_std", "B_std",
        "R_skew", "G_skew", "B_skew", "gray_skew",
        "glcm_contrast", "glcm_dissimilarity", "glcm_homogeneity",
        "glcm_energy", "glcm_correlation", "glcm_asm"
    ]
    return pd.DataFrame([row])[cols]

# Load Model
@st.cache_resource
def load_trained_model():
    model_path = "model_terbaik_klasifikasi_sepatu.joblib"
    if not os.path.exists(model_path):
        return None
    return joblib.load(model_path)

model = load_trained_model()

# Tampilan UI Utama: Minimalis & Sederhana
st.markdown(f"""
<div class="glass-card">
    <div class="title-text">Klasifikasi Citra Rak Sepatu</div>
    <div class="subtitle-text">Unggah foto kondisi rak sepatu (Maks. {MAX_FILE_SIZE_MB}MB) untuk deteksi otomatis</div>
</div>
""", unsafe_allow_html=True)

# Input Upload Foto (Dibatasi 5MB via .streamlit/config.toml & validasi internal)
uploaded_file = st.file_uploader(
    "Pilih foto",
    type=["jpg", "jpeg", "png", "webp"],
    label_visibility="collapsed"
)

if uploaded_file is not None:
    # Validasi ukuran file di sisi aplikasi
    file_size_mb = uploaded_file.size / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        st.markdown(f"""
        <div class="result-card" style="border-color: rgba(235, 87, 87, 0.4);">
            <div style="color: rgba(255, 140, 140, 0.95); font-size: 15px; font-weight: 600;">
                ⚠️ Ukuran file ({file_size_mb:.2f}MB) melebihi batas maksimal {MAX_FILE_SIZE_MB}MB
            </div>
            <div class="result-subtext">Silakan unggah foto dengan ukuran file di bawah {MAX_FILE_SIZE_MB}MB.</div>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    # Buka gambar
    image = Image.open(uploaded_file)
    st.image(image, use_container_width=True)

    if model is None:
        st.error("File model 'model_terbaik_klasifikasi_sepatu.joblib' tidak ditemukan.")
    else:
        # Ekstraksi fitur & klasifikasi langsung
        with st.spinner(""):
            features_df = extract_features(image)
            prediction = model.predict(features_df)[0]
            confidence = 0.0
            if hasattr(model, "predict_proba"):
                probas = model.predict_proba(features_df)[0]
                confidence = float(np.max(probas)) * 100

        # Tentukan status class badge
        pred_clean = prediction.lower().strip()
        badge_class = f"status-{pred_clean}" if pred_clean in ["berantakan", "rapi", "kosong"] else "status-rapi"
        label_display = prediction.upper()

        st.markdown(f"""
        <div class="result-card">
            <div style="color: rgba(255, 255, 255, 0.42); font-size: 12px; letter-spacing: 1.5px; text-transform: uppercase; margin-bottom: 6px;">Hasil Klasifikasi</div>
            <div class="result-badge {badge_class}">{label_display}</div>
            <div class="result-subtext">Tingkat Keyakinan Model: <strong style="color: rgba(255, 255, 255, 0.75);">{confidence:.1f}%</strong></div>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    # Hanya panggil stcli jika TIDAK sedang dijalankan di dalam Streamlit runtime
    try:
        from streamlit import runtime
        is_streamlit_running = runtime.exists()
    except Exception:
        is_streamlit_running = False

    if not is_streamlit_running:
        from streamlit.web import cli as stcli
        sys.argv = ["streamlit", "run", __file__]
        sys.exit(stcli.main())
