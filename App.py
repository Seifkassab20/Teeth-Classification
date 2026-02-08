import streamlit as st
import torch
import json
from PIL import Image
import torchvision.transforms as transforms

# Import models
from models.CNN import CNN
from models.pretrained import PretrainedModel

# -----------------------------
# Load class names
# -----------------------------
try:
    with open("class_names.json", "r") as f:
        class_names = json.load(f)
except FileNotFoundError:
    st.error("Class names file 'class_names.json' not found. Please ensure the file exists.")
    st.stop()

NUM_CLASSES = len(class_names)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Load models
# -----------------------------
@st.cache_resource  # caches model loading
def load_models():
    custom_model = CNN(NUM_CLASSES)
    try:
        custom_model.load_state_dict(torch.load("weights/custom_cnn.pth", map_location=device))
    except FileNotFoundError:
        st.error("Weights file 'weights/custom_cnn.pth' not found. Please ensure the file exists.")
        return None, None
    custom_model.eval()

    medical_model = PretrainedModel(NUM_CLASSES)
    try:
        medical_model.load_state_dict(torch.load("weights/medical_pretrained.pth", map_location=device))
    except FileNotFoundError:
        st.error("Weights file 'weights/medical_pretrained.pth' not found. Please ensure the file exists.")
        return None, None
    medical_model.eval()

    return custom_model, medical_model

custom_model, medical_model = load_models()

# -----------------------------
# Image preprocessing
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🦷 Teeth Image Classification System")
st.write("Upload a dental image and select a model to classify it.")

model_choice = st.selectbox(
    "Select Model",
    ["Custom CNN (From Scratch)", "Medical Pretrained DenseNet"]
)

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Input Image", use_column_width=True)

    img_tensor = transform(image).unsqueeze(0).to(device)

    # Select model
    model = custom_model if model_choice == "Custom CNN (From Scratch)" else medical_model

    if model is None:
        st.error("Selected model could not be loaded. Please check model weights.")
        st.stop()

    # Predict
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        pred_idx = torch.argmax(probs, 1).item()
        confidence = probs[0][pred_idx].item()

    # Display results
    st.subheader("Prediction Result")
    st.write(f"**Model Used:** {model_choice}")
    st.write(f"**Predicted Class:** {class_names[pred_idx]}")
    st.write(f"**Confidence:** {confidence:.2%}")
