import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageClassification
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# ============================================================
# Configuration de la page
# ============================================================
st.set_page_config(
    page_title="AutoDamage AI — Detection de dommages",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CSS personnalise pour un design moderne
# ============================================================
st.markdown("""
<style>
    /* Fond principal */
    .main {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    }
    
    /* Header personnalise */
    .hero-header {
        background: linear-gradient(135deg, #0A8F6E 0%, #0d4f3c 100%);
        padding: 2.5rem 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(10, 143, 110, 0.3);
        color: white;
    }
    .hero-header h1 {
        color: white;
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    .hero-header p {
        color: rgba(255, 255, 255, 0.9);
        margin-top: 0.5rem;
        font-size: 1.05rem;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(255, 255, 255, 0.2);
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        letter-spacing: 2px;
        font-weight: 600;
        margin-bottom: 12px;
    }
    
    /* Cartes de resultats */
    .result-card {
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        margin-bottom: 1rem;
        border-left: 5px solid #0A8F6E;
    }
    .result-card-damaged {
        border-left-color: #dc3545;
        background: linear-gradient(135deg, #fff5f5 0%, #ffffff 100%);
    }
    .result-card-whole {
        border-left-color: #28a745;
        background: linear-gradient(135deg, #f0fff4 0%, #ffffff 100%);
    }
    .result-title {
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #64748b;
        margin-bottom: 0.5rem;
    }
    .result-value {
        font-size: 2rem;
        font-weight: 800;
        margin: 0;
    }
    .result-value-damaged { color: #dc3545; }
    .result-value-whole { color: #28a745; }
    
    /* Section header */
    .section-header {
        font-size: 1.3rem;
        font-weight: 700;
        color: #1e293b;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e2e8f0;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
    }
    [data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }
    .sidebar-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .sidebar-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        opacity: 0.6;
        margin-bottom: 4px;
    }
    .sidebar-value {
        font-size: 1rem;
        font-weight: 600;
    }
    
    /* Indicateur de confiance */
    .confidence-pill {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
        margin: 0.5rem 0;
    }
    .confidence-high { background: #d4edda; color: #155724; }
    .confidence-medium { background: #fff3cd; color: #856404; }
    .confidence-low { background: #f8d7da; color: #721c24; }
    
    /* Disclaimer */
    .disclaimer-box {
        background: linear-gradient(135deg, #fff5f5 0%, #fed7d7 100%);
        border-left: 5px solid #c53030;
        padding: 1.25rem 1.5rem;
        border-radius: 12px;
        margin-top: 2rem;
        color: #742a2a;
    }
    .disclaimer-box strong { color: #c53030; }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 1.5rem;
        color: #94a3b8;
        font-size: 0.85rem;
        margin-top: 2rem;
        border-top: 1px solid #e2e8f0;
    }
    
    /* Cacher le menu Streamlit par defaut */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Bouton upload personnalise */
    [data-testid="stFileUploaderDropzone"] {
        background: rgba(10, 143, 110, 0.05);
        border: 2px dashed #0A8F6E;
        border-radius: 16px;
        padding: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# Modele et preprocessing
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "./model_weights"

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

class ClassifierWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        return self.model(pixel_values=x).logits

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

# ============================================================
# Helpers
# ============================================================
def get_confidence_class(confidence):
    if confidence >= 85:
        return "confidence-high", "Confiance elevee", "🟢"
    elif confidence >= 65:
        return "confidence-medium", "Confiance moderee", "🟡"
    else:
        return "confidence-low", "Confiance faible", "🔴"

def is_damaged_class(label):
    return "damage" in label.lower() and "not" not in label.lower() and "whole" not in label.lower()

# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    st.markdown("# 🚗 AutoDamage AI")
    st.markdown("*Pre-evaluation de sinistre automobile par IA*")
    st.markdown("---")
    
    st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-label">Modele</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-value">MobileNet-V2</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-label">Classes detectees</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sidebar-value">{", ".join(id2label.values())}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-label">Device</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sidebar-value">{str(DEVICE).upper()}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 📊 Performance")
    st.markdown("**Precision** : ~90% (test set)")
    st.markdown("**Temps d'inference** : < 1s")
    
    st.markdown("---")
    st.markdown("### 🔒 Confidentialite")
    st.caption("Les images uploadees ne sont pas stockees. Tout est traite en memoire.")

# ============================================================
# Hero header
# ============================================================
st.markdown("""
<div class="hero-header">
    <div class="hero-badge">INTELLIGENCE ARTIFICIELLE</div>
    <h1>🚗 Detecteur de dommages vehicules</h1>
    <p>Pre-evaluation automatique de sinistres automobiles a partir d'une photo</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# Section "Comment ca marche ?"
# ============================================================
with st.expander("ℹ️ Comment ca marche ?", expanded=False):
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("### 📸 1. Upload")
        st.markdown("Vous envoyez une photo de vehicule au format JPG ou PNG.")
    with col_b:
        st.markdown("### 🧠 2. Analyse")
        st.markdown("Le modele **MobileNet-V2** (pre-entraine sur 14M d'images) analyse la photo.")
    with col_c:
        st.markdown("### 🎯 3. Resultat")
        st.markdown("Vous obtenez une prediction avec score de confiance et heatmap GradCAM.")
    
    st.markdown("---")
    st.markdown("""
    **Comment lire les resultats** :
    - 🟢 **Confiance ≥ 85%** : prediction tres fiable
    - 🟡 **Confiance 65-85%** : prediction probable, a verifier avec la heatmap
    - 🔴 **Confiance < 65%** : le modele hesite, expertise humaine recommandee
    
    **Heatmap GradCAM** : les zones rouges/oranges montrent ou le modele "regarde" pour
    decider. Si la heatmap se concentre sur la zone endommagee, la prediction est fiable.
    Si elle pointe ailleurs (arriere-plan, ciel...), la prediction est suspecte.
    """)

# ============================================================
# Upload
# ============================================================
st.markdown('<div class="section-header">📤 Uploader une image</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Glissez-deposez une image ou cliquez pour parcourir",
    type=["jpg", "jpeg", "png"],
    help="Formats acceptes : JPG, JPEG, PNG — max 200 MB",
    label_visibility="collapsed"
)

# ============================================================
# Analyse de l'image
# ============================================================
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.markdown('<div class="section-header">🔍 Resultats de l\'analyse</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("##### 📷 Image originale")
        st.image(image, use_container_width=True)
        st.caption(f"Fichier : `{uploaded_file.name}` • Dimensions : {image.size[0]} × {image.size[1]} px")

    with col2:
        with st.spinner("🔄 Analyse en cours..."):
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

        # ----- Carte de prediction -----
        is_dmg = is_damaged_class(pred_label)
        card_class = "result-card-damaged" if is_dmg else "result-card-whole"
        value_class = "result-value-damaged" if is_dmg else "result-value-whole"
        icon = "⚠️" if is_dmg else "✅"
        status_text = "Dommage detecte" if is_dmg else "Vehicule intact"
        
        st.markdown(f"""
        <div class="result-card {card_class}">
            <div class="result-title">Diagnostic</div>
            <p class="result-value {value_class}">{icon} {status_text}</p>
            <p style="color: #64748b; margin: 0;">Classe predite : <code>{pred_label}</code></p>
        </div>
        """, unsafe_allow_html=True)

        # ----- Indicateur de confiance -----
        conf_class, conf_label, conf_emoji = get_confidence_class(confidence)
        st.markdown(f"""
        <div class="confidence-pill {conf_class}">
            {conf_emoji} {conf_label} — {confidence:.1f}%
        </div>
        """, unsafe_allow_html=True)
        
        st.progress(int(confidence) / 100)
        
        if confidence >= 85:
            st.success("✓ Le modele est sur de lui — prediction fiable.")
        elif confidence >= 65:
            st.warning("⚠ Confiance moderee — verifiez la heatmap ci-dessous.")
        else:
            st.error("✗ Confiance faible — expertise humaine vivement recommandee.")

        # ----- Probabilites detaillees -----
        st.markdown("##### 📊 Probabilites par classe")
        for i, p in enumerate(probs):
            label_name = id2label[i]
            is_pred = (i == pred_idx)
            marker = "**▸**" if is_pred else "  "
            st.markdown(f"{marker} `{label_name}`")
            st.progress(float(p), text=f"{p*100:.1f}%")

    # ----- GradCAM en pleine largeur -----
    st.markdown('<div class="section-header">🔥 Heatmap GradCAM — Zones d\'attention</div>', unsafe_allow_html=True)
    
    col_g1, col_g2 = st.columns([2, 1])
    with col_g1:
        st.image(gradcam_overlay, use_container_width=True)
    with col_g2:
        st.markdown("""
        **Comment lire cette heatmap ?**
        
        Les zones colorees (rouge → jaune) indiquent ou le modele
        a "regarde" pour prendre sa decision.
        
        🔴 **Rouge** : forte attention  
        🟡 **Jaune** : attention moderee  
        🔵 **Bleu** : peu d'attention  
        
        **Verification** : si la heatmap pointe vers la zone
        endommagee, la prediction est fiable. Si elle pointe
        ailleurs (arriere-plan, ciel...), prediction suspecte.
        """)

    # ----- Disclaimer -----
    st.markdown("""
    <div class="disclaimer-box">
        <strong>⚠️ Pre-evaluation automatique</strong><br>
        Cet outil est une <strong>aide a la decision</strong>. L'avis d'un expert reste
        <strong>obligatoire</strong> pour la validation du sinistre. Ne pas utiliser cette
        prediction comme seule base pour une decision financiere.
    </div>
    """, unsafe_allow_html=True)

else:
    # Etat vide
    st.info("👆 **Uploadez une image** pour commencer l'analyse — le modele est pret.")
    
    # Exemples de cas d'usage
    st.markdown('<div class="section-header">💡 Cas d\'usage</div>', unsafe_allow_html=True)
    col_x, col_y, col_z = st.columns(3)
    with col_x:
        st.markdown("""
        #### 📋 Declaration de sinistre
        Acceleration du traitement des declarations en pre-classifiant
        automatiquement les photos envoyees par les assures.
        """)
    with col_y:
        st.markdown("""
        #### 🔍 Detection de fraude
        Identification des photos suspectes ou ne montrant pas
        de dommages reels avant validation par un expert.
        """)
    with col_z:
        st.markdown("""
        #### ⚡ Triage rapide
        Priorisation des dossiers : les cas evidents sont traites
        rapidement, les cas complexes vont aux experts.
        """)

# ============================================================
# Footer
# ============================================================
st.markdown("""
<div class="footer">
    Made by Malo Héry & Jules Boiziau using <strong>Streamlit</strong> & <strong>PyTorch</strong>
    • Modele <strong>MobileNet-V2</strong> fine-tune sur Car Damage Dataset
    </div>
""", unsafe_allow_html=True)
