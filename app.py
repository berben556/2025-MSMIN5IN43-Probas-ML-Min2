import streamlit as st
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert import HTMLExporter
import os
import torch
from PIL import Image
from torchvision import transforms
import sys
import torch.nn.functional as F
import cv2
import numpy as np

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
# Grad-CAM Class
# -------------------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx=None):
        self.model.zero_grad()
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        loss = output[:, class_idx]
        loss.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1)

        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        cam = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))
        cam = cam - cam.min()
        cam = cam / cam.max()

        return cam

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
        # Grad-CAM
        st.markdown("### Grad-CAM (Carte de chaleur)")
        grad_cam = GradCAM(model, model.layer4[-1])
        cam = grad_cam.generate(input_tensor, pred)

        # Superposer la heatmap sur l'image
        img_np = np.array(image.resize((224, 224)))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        superimposed_img = heatmap + np.float32(img_np) / 255
        superimposed_img = superimposed_img / np.max(superimposed_img)
        superimposed_img = np.uint8(255 * superimposed_img)

        st.image(superimposed_img, caption="Image avec Grad-CAM", use_column_width=True)
        st.warning(
            "Cet outil est une aide à la décision et ne remplace pas un diagnostic médical."
        )
