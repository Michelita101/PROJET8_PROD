# ===== Imports =====
import streamlit as st
from PIL import Image
from pathlib import Path
import requests
import io
import base64

# ===== Config =====
st.set_page_config(
    page_title="definitiveDRIVE",
    layout="wide"
)

API_URL = "http://api:8000"

# ===== Paths =====
BASE_DIR = Path(__file__).resolve().parent
LOGO_PATH = BASE_DIR / "assets" / "logo.png"

# ===== Header =====
col1, col2 = st.columns([1, 4])

with col1:
    st.image(str(LOGO_PATH), width=120)

with col2:
    st.title("definitiveDRIVE")
    st.markdown("_Semantic segmentation for autonomous driving_")

st.divider()

st.success("Streamlit app ready! Next step: image selection ;-)")

# ==== Fonctions API ====
@st.cache_data
def get_available_images():
    response = requests.get(f"{API_URL}/images")
    response.raise_for_status()
    return response.json()["images"]

@st.cache_data
def get_image(image_id):
    response = requests.get(f"{API_URL}/image/{image_id}")
    response.raise_for_status()
    return Image.open(io.BytesIO(response.content))

@st.cache_data
def get_real_mask(image_id):
    response = requests.get(f"{API_URL}/mask-real/{image_id}")
    response.raise_for_status()
    return Image.open(io.BytesIO(response.content))

def get_predicted_mask(image_id):
    response = requests.get(f"{API_URL}/predict/{image_id}")
    response.raise_for_status()
    return Image.open(io.BytesIO(response.content))

# ==== Sidebar ====
st.sidebar.header("🔍 Image selection")

image_ids = get_available_images()

selected_image_id = st.sidebar.selectbox(
    "Choose a scene",
    image_ids
)

run_segmentation = st.sidebar.button("🚗 Run segmentation")

# ==== Feedback utilisateur ====
st.info(f"Selected image: **{selected_image_id}**")


# ==== Déclencher l'affichage au clic ====
if run_segmentation:
    with st.spinner("Running semantic segmentation..."):
        real_image = get_image(selected_image_id)
        real_mask = get_real_mask(selected_image_id)
        predicted_mask = get_predicted_mask(selected_image_id)

    st.success("Segmentation completed!")

    # ==== Affichage des résultats ====
    col_img, col_gt, col_pred = st.columns(3)

    with col_img:
        st.subheader("📷 Real image")
        st.image(real_image, use_container_width=True)

    with col_gt:
        st.subheader("🎯 Ground truth mask")
        st.image(real_mask, use_container_width=True)

    with col_pred:
        st.subheader("🤖 Predicted mask")
        st.image(predicted_mask, use_container_width=True)


