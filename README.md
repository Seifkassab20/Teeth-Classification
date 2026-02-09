# 🦷 Dental AI Diagnostician

## 📋 Overview

**Dental AI Diagnostician** is a professional deep learning application designed to automate the classification of dental conditions from images. Built with **PyTorch** and **Streamlit**, this tool aids dental professionals by providing a second opinion with fast, consistent, and visual diagnostic feedback.

The system leverages the power of both custom-designed architectures and state-of-the-art transfer learning to identify various oral pathologies with high accuracy.

## ✨ Features

- **Dual AI Engines**:
  - **Custom CNN**: A specialized, lightweight deep learning model built from scratch.
  - **Pretrained ResNet50**: A robust transfer learning model fine-tuned for medical imaging.
- **Professional Dashboard**: A clean, modern UI featuring a sidebar control panel and prediction cards.
- **Detailed Analytics**:
  - Color-coded confidence indicators (Green/Yellow/Red).
  - Full probability distribution charts to understand model certainty.
- **Production Ready**: Fully containerized with **Docker** and NVIDIA GPU support.

## 🧠 Model Architectures

### 1. Custom CNN (From Scratch)

A specialized 4-block Convolutional Neural Network designed to capture specific dental textures and patterns.

- **Structure**: 4x (Conv2D -> BatchNorm -> ReLU -> MaxPool)
- **Regularization**: Dropout (0.5) to prevent overfitting.
- **Optimization**: Trained with Adam optimizer and CrossEntropyLoss.

### 2. Medical Pretrained ResNet50

Uses a ResNet50 backbone pretrained on ImageNet, with layers unfrozen and fine-tuned specifically for this dental dataset. This model excels at identifying high-level semantic features.

## 🧬 Supported Classes

The model is trained to classify the following 7 dental conditions:

- **CaS** (Caries / Cavities)
- **CoS** (Composite Restoration)
- **Gum** (Gingivitis / Gum Disease)
- **MC** (Metal Crown)
- **OC** (Orthodontic Classification)
- **OLP** (Oral Lichen Planus)
- **OT** (Other / Tumors)

## 🚀 Installation & Usage

### Option 1: Running Locally

1. **Clone the repository**

   ```bash
   git clone <repository_url>
   cd Teeth-Classification
   ```

2. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application**
   ```bash
   streamlit run App.py
   ```

### Option 2: Running with Docker (Recommended)

This method ensures all dependencies and drivers are correctly isolated.

1. **Build the Docker Image** (with GPU support)

   ```bash
   docker build -t dental-ai-app-gpu .
   ```

2. **Run the Container**

   ```bash
   docker run --gpus all -p 8501:8501 dental-ai-app-gpu
   ```

3. **Access the App**
   Open your browser and navigate to: `http://localhost:8501`

## 📂 Project Structure

```
Teeth-Classification/
├── App.py                 # Main Streamlit application
├── Dockerfile             # Container configuration
├── requirements.txt       # Python dependencies
├── class_names.json       # JSON list of class labels
├── models/
│   ├── cnn.py             # Custom CNN architecture definition
│   ├── pretrained.py      # ResNet50 model definition
│   ├── train_cnn.py       # Training script for Custom CNN
│   └── train_pretrained.py# Training script for Pretrained model
├── dataloader/            # Data loading utilities
└── weights/               # Saved model weights (.pth files)
```

## 🛠️ Technology Stack

- **Languages**: Python
- **ML Framework**: PyTorch
- **Web Framework**: Streamlit
- **DevOps**: Docker, NVIDIA Container Toolkit
- **Data Processing**: Pandas, Pillow, Torchvision
