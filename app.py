import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Vehicle Detection & Classification",
    page_icon="🚗",
    layout="wide"
)

# Load models
@st.cache_resource
def load_models():
    try:
        # Load detection model
        detection_model = YOLO('yolov8s_vehicle_detection.pt')
        
        # Load classification model structure
        classification_model = models.resnet18(pretrained=False)
        num_features = classification_model.fc.in_features
        classification_model.fc = nn.Linear(num_features, 4)
        
        # Load weights
        state_dict = torch.load('resnet18_classification.pth', map_location=torch.device('cpu'))
        classification_model.load_state_dict(state_dict)
        classification_model.eval()
        
        return detection_model, classification_model, True
    except Exception as e:
        return None, None, False

# Preprocessing transforms
cls_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

st.title("🚗 Vehicle Detection & Classification System")
detection_model, classification_model, models_loaded = load_models()

if not models_loaded:
    st.error("❌ Model files missing! Ensure .pt and .pth files are in the root directory.")
    st.stop()

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Run Detection
    results = detection_model.predict(source=image, conf=0.5)
    result = results[0]
    detections = result.boxes.data.cpu().numpy()
    
    if len(detections) > 0:
        # Run Classification for each detection
        classes_list = ['Car', 'Bus', 'Truck', 'Motorcycle']
        
        st.write("### 🎯 Detection & Classification Results")
        annotated_img = Image.fromarray(result.plot()[..., ::-1])
        st.image(annotated_img, caption="Detected Vehicles")
    else:
        st.warning("No vehicles detected.")
