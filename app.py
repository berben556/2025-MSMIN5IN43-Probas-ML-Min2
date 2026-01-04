import streamlit as st
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert import HTMLExporter
import os
import torch
from PIL import Image
from torchvision import transforms
import sys

# -------------------------
# Config notebooks
# -------------------------
NOTEBOOK_DIR = "notebooks"

def run_notebook(notebook_name):
    path = os.path.join(NOTEBOOK_DIR, notebook_name)
    with open(path) as f:
        nb = nbformat.read(f, as_version=4)
    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
    ep.preprocess(nb, {'metadata': {'path': NOTEBOOK_DIR}})
    return True

# -------------------------
# Config modèle
# -------------------------
sys.path.append(".")
from src.models.resnet import get_model

CLASSES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

@st.cache_resource
def load_model(version=1):
    model = get_model(num_classes=4)
    model.load_state_dict(torch.load("experiments/model.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model(version=2)

# -------------------------
# UI
# -------------------------
st.title("Tumor Detection Project")

tab1, tab2 = st.tabs(["Notebooks", "Test du modèle"])

# ==========================================================
# TAB 1 — NOTEBOOKS
# ==========================================================
with tab1:
    st.subheader("Exploration et démonstrations")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Exploration Dataset"):
            with st.spinner("Running dataset..."):
                run_notebook("01_exploration_dataset.ipynb")
            st.image("outputs/dataset.png", caption="Dataset Visualization")

    with col2:
        if st.button("Data Analysis"):
            with st.spinner("Running analysis..."):
                run_notebook("02_data_analysis.ipynb")
            st.image("outputs/analysis.png", caption="Analysis Visualization")

    with col3:
        if st.button("Grad-CAM Demo"):
            with st.spinner("Running gradcam..."):
                run_notebook("03_gradcam_demo.ipynb")
            st.image("outputs/gradcam_result.png", caption="Grad-CAM Result")

# ==========================================================
# TAB 2 — TEST RÉEL DU MODÈLE
# ==========================================================
with tab2:
    st.subheader("Test du modèle sur une image IRM")
    st.write("Upload d’une image IRM jamais vue par le modèle")

    uploaded_file = st.file_uploader(
        "Choisir une image IRM",
        type=["jpg", "png", "jpeg"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Image IRM", use_column_width=True)

        input_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)
            pred = probs.argmax(1).item()

        st.markdown("### Résultat")
        st.success(f"Tumeur détectée : **{CLASSES[pred]}**")

        st.markdown("### Probabilités")
        for i, cls in enumerate(CLASSES):
            st.write(f"{cls} : {probs[0][i]:.2f}")

        st.warning(
            "Cet outil est une aide à la décision et ne remplace pas un diagnostic médical."
        )
