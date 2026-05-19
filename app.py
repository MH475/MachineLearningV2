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
MODEL_PATH = "./model_weights"

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
        st.error(f"Modele introuvable a {MODEL_PATH}.")
        st.stop()
    model = AutoModelForImageClassification.from_pretrained(MODEL_PATH)
    model.to(DEVICE)
    model.eval()
    return model

model = load_model()
id2label = model.config.id2label

# ------------------------------------------------------------------
# Helper : indicateur visuel de confiance (point 1)
# ------------------------------------------------------------------
def get_confidence_info(confidence):
    """Retourne (emoji, niveau, couleur) selon le score de confiance."""
    if confidence >= 85:
        return "🟢", "Confiance elevee", "#28a745"
    elif confidence >= 65:
        return "🟡", "Confiance moderee", "#ffc107"
    else:
        return "🔴", "Confiance faible", "#dc3545"

# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------
st.sidebar.title("A propos")
st.sidebar.info(
    "Pre-evaluation automatique — l'avis d'un expert reste necessaire "
    "pour la validation du sinistre."
)
st.sidebar.markdown("---")
st.sidebar.markdown("**Modele :** MobileNet-V2")
st.sidebar.markdown(f"**Classes :** {', '.join(id2label.values())}")
st.sidebar.markdown(f"**Device :** {DEVICE}")

# ------------------------------------------------------------------
# Page principale
# ------------------------------------------------------------------
st.title("🚗 Detecteur de dommages vehicules")
st.markdown(
    "Uploadez une image pour obtenir une prediction "
    "avec score de confiance et visualisation GradCAM."
)

# Section "Comment ca marche ?" (point 3)
with st.expander("ℹ️ Comment ca marche ?"):
    st.markdown("""
    **Objectif** : aider les experts en assurance a pre-evaluer rapidement si un vehicule
    presente des dommages visibles, a partir d'une simple photo.

    **Modele utilise** : MobileNet-V2, un reseau de neurones convolutif (CNN)
    pre-entraine sur ImageNet (14 millions d'images) puis affine sur un dataset
    de photos de vehicules endommages et intacts.

    **Comment lire les resultats** :
    - 🟢 **Confiance elevee (≥ 85%)** : la prediction est tres probable
    - 🟡 **Confiance moderee (65 - 85%)** : la prediction est probable mais a verifier
    - 🔴 **Confiance faible (< 65%)** : le modele hesite, expertise humaine recommandee

    **Heatmap GradCAM** : les zones rouge/orange indiquent ou le modele a "regarde"
    pour prendre sa decision. Si la heatmap se concentre sur la zone endommagee,
    la prediction est fiable. Si elle pointe ailleurs (arriere-plan, ciel...),
    la prediction doit etre remise en question.

    **Limites** :
    - Le modele a ete entraine sur des photos standard. Des images floues, sombres
      ou prises sous un angle inhabituel peuvent reduire la fiabilite.
    - Le modele ne quantifie pas le cout des reparations, il indique seulement
      la presence de dommages visibles.
    - Cette application est un outil d'aide a la decision et ne remplace en
      aucun cas l'expertise d'un professionnel.
    """)

# ------------------------------------------------------------------
# Upload d'image
# ------------------------------------------------------------------
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

        # ----- Affichage de la prediction -----
        is_damaged = "damage" in pred_label.lower() and "not" not in pred_label.lower() and "whole" not in pred_label.lower()
        icon = "🔴" if is_damaged else "🟢"
        st.metric(
            label="Prediction",
            value=f"{icon} {pred_label.upper()}",
            delta=f"{confidence:.1f}% de confiance"
        )

        # ----- Indicateur visuel de confiance (point 1) -----
        conf_emoji, conf_level, conf_color = get_confidence_info(confidence)
        st.markdown(f"**Niveau de confiance** : {conf_emoji} {conf_level}")
        st.progress(int(confidence), text=f"{confidence:.1f}%")

        # Message contextuel selon la confiance
        if confidence >= 85:
            st.success("Le mo
