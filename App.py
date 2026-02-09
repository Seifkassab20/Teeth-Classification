import streamlit as st
import torch
import torch.nn.functional as F
import pandas as pd
import json
from PIL import Image
import torchvision.transforms as transforms

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Dental AI Diagnostician",
    page_icon="🦷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a professional look
st.markdown("""
<style>
    .main {
        background-color: #fafafa;
    }
    h1 {
        color: #2c3e50;
        font-family: 'Helvetica Neue', sans-serif;
    }
    .stButton>button {
        width: 100%;
        background-color: #007bff;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #0056b3;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .prediction-card {
        background-color: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        margin-top: 1rem;
        text-align: center;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2c3e50;
    }
    .metric-label {
        color: #7f8c8d;
        font-size: 1rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. MODEL & DATA LOADING
# -----------------------------------------------------------------------------
from models.cnn import CNN
from models.pretrained import PretrainedModel

# Load Class Names
try:
    with open("class_names.json", "r") as f:
        class_names = json.load(f)
except FileNotFoundError:
    st.error("🚨 Class names file `class_names.json` not found.")
    st.stop()

NUM_CLASSES = len(class_names)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_models():
    # Initialize models
    custom_model = CNN(NUM_CLASSES).to(device)
    medical_model = PretrainedModel(NUM_CLASSES).to(device)

    # Load Weights
    # 1. Custom CNN
    try:
        custom_model.load_state_dict(torch.load("weights/custom_cnn.pth", map_location=device))
        custom_model.eval()
    except FileNotFoundError:
        st.warning("⚠️ Weights for `Custom CNN` not found.")
        custom_model = None

    # 2. Pretrained Model
    try:
        medical_model.load_state_dict(torch.load("weights/medical_pretrained.pth", map_location=device))
        medical_model.eval()
    except FileNotFoundError:
        st.warning("⚠️ Weights for `Medical Pretrained Model` not found.")
        medical_model = None

    return custom_model, medical_model

custom_model, medical_model = load_models()

# Image Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
    # Note: No Normalization, matching the training pipeline
])

# -----------------------------------------------------------------------------
# 3. SIDEBAR CONTROLS
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Control Panel")
    
    st.markdown("### Model Selection")
    model_choice = st.radio(
        "Choose Inference Model:",
        ["Custom CNN (From Scratch)", "Medical Pretrained ResNet50"],
        index=0,
        help="Select which AI architecture to use for prediction."
    )
    
    st.markdown("---")
    st.markdown("### About")
    st.info(
        "This system uses Deep Learning to classify dental conditions from X-rays and intraoral images. "
        "Upload a clear image for the best results."
    )
    
    st.markdown("---")
    st.caption(f"Running on: **{device.type.upper()}**")

# -----------------------------------------------------------------------------
# 4. MAIN INTERFACE
# -----------------------------------------------------------------------------
st.title("🦷 Dental Diagnosis AI")
st.markdown("#### Automated classification of dental conditions")

col1, col2 = st.columns([1, 1.2], gap="large")

with col1:
    st.markdown("### 1. Upload Image")
    uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Patient Image", use_column_width=True)

with col2:
    st.markdown("### 2. Analysis Results")
    
    if uploaded_file:
        model = custom_model if "Custom" in model_choice else medical_model
        
        if model is None:
            st.error("Selected model is not available.")
        else:
            # Prediction Logic
            img_tensor = transform(image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                outputs = model(img_tensor)
                probs = F.softmax(outputs, dim=1)
                
                # Get Top Prediction
                conf, pred_idx = torch.max(probs, 1)
                prediction = class_names[pred_idx.item()]
                confidence = conf.item()
                
                # Colors based on confidence
                color = "#28a745" if confidence > 0.8 else "#ffc107" if confidence > 0.5 else "#dc3545"

                # Display Main Result Card
                st.markdown(f"""
                <div class="prediction-card" style="border-top: 5px solid {color};">
                    <div class="metric-label">Diagnosis</div>
                    <div class="metric-value">{prediction}</div>
                    <div style="margin-top: 10px; color: {color}; font-weight: bold;">
                        Confidence: {confidence:.2%}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Detailed Metrics
                st.markdown("#### Probability Distribution")
                
                # Prepare data for chart
                probs_np = probs.cpu().numpy().flatten()
                df = pd.DataFrame({
                    "Condition": class_names,
                    "Probability": probs_np
                }).sort_values(by="Probability", ascending=False)

                # Display Chart
                st.bar_chart(df.set_index("Condition"), color=color)

                # Show Raw Data Expander
                with st.expander("View Raw Probabilities"):
                    st.dataframe(df.style.format({"Probability": "{:.2%}"}))

    else:
        st.info("👈 Please upload an image to begin analysis.")
