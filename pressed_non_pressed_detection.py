import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageOps
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module='torchvision.models._utils')

# Define the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load the pre-trained model and modify the final layer
def load_model(weights_path):
    model_ft = models.resnet18(weights=None)  # Use weights=None instead of pretrained=False
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 1)  # Binary classification

    # Load the best model weights
    model_ft.load_state_dict(torch.load(weights_path, map_location=device))
    model_ft = model_ft.to(device)
    model_ft.eval()  # Set the model to evaluation mode

    return model_ft

# Define the image preprocessing function
def preprocess_image(pil_image):
    preprocess = transforms.Compose([
        transforms.Resize(224),  # Resize to match model input size
        transforms.CenterCrop(224),  # Center crop to match model input size
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    image = preprocess(pil_image).unsqueeze(0)  # Add batch dimension
    return image

# Define the prediction function
def predict(pil_image, model):
    image = preprocess_image(pil_image)
    image = image.to(device)
    
    with torch.no_grad():
        outputs = model(image)
        prediction = torch.sigmoid(outputs).item()  # Convert logits to probability
        return prediction
