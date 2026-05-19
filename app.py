import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageClassification
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# Configuration de la page
st.set_page_config(
    page_title="Detecteur de dommages vehicules",
    page_icon="🚗",
    layout="wide"
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "./model_weights"  # chemin ABSOLU (corrige le bug Colab)

# Pipeline de preprocessing
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Wrapper pour GradCAM
class ClassifierWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        return self.model(pixel_values=x).logits

# Chargement du modele (cache pour ne le charger qu'une fois)
@st.cache_resource
def load_model():
    import os
    if not os.path.exists(MODEL_PATH):
        st.error(f"Modele introuvable a {MODEL_PATH}. Avez-vous execute la cellule de sauvegarde du modele ?")
        st.stop()
    model = AutoModelForImageClassification.from_pretrained(MODEL_PATH)
    model.to(DEVICE)
    model.eval()
    return model

model = load_model()
id2label = model.config.id2label

# Sidebar
st.sidebar.title("A propos")
st.sidebar.info(
    "Pre-evaluation automatique — l'avis d'un expert reste necessaire "
    "pour la validation du sinistre."
)
st.sidebar.markdown("---")
st.sidebar.markdown("**Modele :** MobileNet-V2")
st.sidebar.markdown(f"**Classes :** {', '.join(id2label.values())}")
st.sidebar.markdown(f"**Device :** {DEVICE}")

# Page principale
st.title("🚗 Detecteur de dommages vehicules")
st.markdown(
    "Uploadez une image pour obtenir une prediction "
    "avec score de confiance et visualisation GradCAM."
)

uploaded_file = st.file_uploader(
    "Choisir une image",
    type=["jpg", "jpeg", "png"],
    help="Formats acceptes : JPG, JPEG, PNG"
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Image originale")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("Analyse")

        with st.spinner("Analyse en cours..."):
            input_tensor = val_transform(image).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                outputs = model(pixel_values=input_tensor)
                logits = outputs.logits
                probs = F.softmax(logits, dim=1)[0].cpu().numpy()

            pred_idx = int(np.argmax(probs))
            pred_label = id2label[pred_idx]
            confidence = float(probs[pred_idx]) * 100

            # GradCAM
            wrapped = ClassifierWrapper(model)
            target_layer = model.mobilenet_v2.conv_1x1.convolution
            cam = GradCAM(model=wrapped, target_layers=[target_layer])
            grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0]

            img_resized = image.resize((224, 224))
            rgb_img = np.array(img_resized).astype(np.float32) / 255.0
            gradcam_overlay = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        is_damaged = "damage" in pred_label.lower() and "not" not in pred_label.lower()
        icon = "🔴" if is_damaged else "🟢"
        st.metric(
            label="Prediction",
            value=f"{icon} {pred_label.upper()}",
            delta=f"{confidence:.1f}% de confiance"
        )

        st.markdown("**Probabilites par classe :**")
        for i, p in enumerate(probs):
            st.progress(float(p), text=f"{id2label[i]} : {p*100:.1f}%")

        st.image(gradcam_overlay, caption="Heatmap GradCAM — zones regardees par le modele",
                 use_container_width=True)

    st.markdown("---")
    st.error(
        "⚠️ Pre-evaluation automatique — l'avis d'un expert reste necessaire "
        "pour la validation du sinistre."
    )
else:
    st.info("👆 Uploadez une image pour commencer l'analyse.")
