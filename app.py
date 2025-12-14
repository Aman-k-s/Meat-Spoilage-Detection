import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt


st.set_page_config(
    page_title="CNN Intensity Analyzer",
    page_icon="🔬",
    layout="wide"
)

#CSS STYLES
st.markdown("""
<style>
.stApp {
    background: linear-gradient(to right, #f8fafc, #eef2ff);
}

h1, h2, h3 {
    color: #1e3a8a;
    font-weight: 700;
}

.info-box {
    background: #e0f2fe;
    padding: 15px;
    border-radius: 10px;
    border-left: 6px solid #0284c7;
    margin-bottom: 15px;
}

.success-box {
    background: #dcfce7;
    padding: 15px;
    border-radius: 10px;
    border-left: 6px solid #16a34a;
}

.image-card {
    background: white;
    padding: 15px;
    border-radius: 14px;
    box-shadow: 0px 4px 16px rgba(0,0,0,0.08);
}
</style>
""", unsafe_allow_html=True)


st.markdown("## 🔬 CNN-based Intensity & Dot Analysis")

st.markdown(
    """
    This application uses a **CNN (ResNet50)** to extract  
    **low-level feature activations** from microscopic images and computes:

    • Mean feature intensity  
    • Number of activation dots  
    • Average dot intensity  
    """
)

st.divider()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.markdown(
    f"<div class='info-box'>🖥️ <b>Using device:</b> {device}</div>",
    unsafe_allow_html=True
)

#Load CNN Model
@st.cache_resource
def load_cnn():
    resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    resnet.eval()
    cnn_layer = nn.Sequential(*list(resnet.children())[:4])
    return cnn_layer.to(device)

cnn_layer = load_cnn()


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def analyze_single_image(img):
    input_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        feature_maps = cnn_layer(input_tensor)

    fmap = feature_maps.squeeze(0).cpu().numpy()
    activation_map = np.mean(fmap, axis=0)

    # Normalize
    activation_map = (activation_map - activation_map.min()) / (
        activation_map.max() - activation_map.min() + 1e-8
    )
    activation_map_8u = (activation_map * 255).astype(np.uint8)

    # Metrics
    mean_intensity = np.mean(activation_map_8u)

    _, binary = cv2.threshold(
        activation_map_8u, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    num_labels, _ = cv2.connectedComponents(binary)
    num_dots = num_labels - 1

    bright_pixels = activation_map_8u[binary == 255]
    avg_dot_intensity = np.mean(bright_pixels) if len(bright_pixels) > 0 else 0

    return activation_map, binary, mean_intensity, num_dots, avg_dot_intensity

#image uploader
uploaded_file = st.file_uploader(
    "📂 Upload a Microscopic Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    img = Image.open(uploaded_file).convert("RGB")

    with st.spinner("Running CNN feature extraction..."):
        activation_map, binary, mean_i, dots, avg_dot_i = analyze_single_image(img)

    st.markdown(
        "<div class='success-box'>✅ Analysis completed successfully</div>",
        unsafe_allow_html=True
    )

    st.divider()

    st.subheader("📊 Quantitative Results")

    c1, c2, c3 = st.columns(3)

    c1.metric("Mean Intensity (Whole Image)", f"{mean_i:.2f}")
    c2.metric("Number of Dots", dots)
    c3.metric("Average Dot Intensity", f"{avg_dot_i:.2f}")

    st.divider()


    st.subheader("🖼️ Visual Interpretation")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("<div class='image-card'>", unsafe_allow_html=True)
        st.image(img, caption="Original Image", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='image-card'>", unsafe_allow_html=True)
        fig, ax = plt.subplots()
        ax.imshow(activation_map, cmap="viridis")
        ax.set_title("CNN Feature Intensity")
        ax.axis("off")
        st.pyplot(fig)
        st.markdown("</div>", unsafe_allow_html=True)

    with col3:
        st.markdown("<div class='image-card'>", unsafe_allow_html=True)
        fig, ax = plt.subplots()
        ax.imshow(binary, cmap="gray")
        ax.set_title(f"Detected Dots: {dots}")
        ax.axis("off")
        st.pyplot(fig)
        st.markdown("</div>", unsafe_allow_html=True)
