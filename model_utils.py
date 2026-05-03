"""
Cloud-Enabled AI System for Skin Cancer Detection — Model Utilities
Handles EfficientNet-B0 model loading, image preprocessing, and prediction logic
for the HAM10000 skin cancer classification task using Explainable Federated Learning.
"""

import os
import io
import base64
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# ============================================================
# Class Metadata — The 7 HAM10000 Diagnostic Categories
# Standard alphabetical mapping used during training
# ============================================================

CLASS_INFO = {
    0: {
        "code": "akiec",
        "name": "Actinic Keratosis",
        "full_name": "Actinic Keratoses & Intraepithelial Carcinoma (Bowen's Disease)",
        "risk": "high",
        "description": "A pre-cancerous scaly patch caused by years of sun exposure. Bowen's disease is an early form of squamous cell carcinoma confined to the epidermis.",
        "recommendation": "Consult a dermatologist promptly. These lesions can progress to invasive squamous cell carcinoma if left untreated. Treatment options include cryotherapy, topical chemotherapy, or surgical excision."
    },
    1: {
        "code": "bcc",
        "name": "Basal Cell Carcinoma",
        "full_name": "Basal Cell Carcinoma",
        "risk": "high",
        "description": "The most common type of skin cancer. It arises from basal cells in the deepest layer of the epidermis. Usually appears as a pearly or waxy bump, or a flat, flesh-colored lesion.",
        "recommendation": "Seek immediate dermatological evaluation. While BCC rarely metastasizes, it can cause significant local tissue destruction. Mohs surgery is the gold standard treatment."
    },
    2: {
        "code": "bkl",
        "name": "Benign Keratosis",
        "full_name": "Benign Keratosis-like Lesions (Seborrheic Keratosis, Solar Lentigo)",
        "risk": "low",
        "description": "A non-cancerous skin growth that appears as a waxy, scaly, slightly elevated patch. Common in older adults. Includes seborrheic keratoses, solar lentigines, and lichen-planus-like keratoses.",
        "recommendation": "Generally harmless and requires no treatment. If the lesion changes in appearance, bleeds, or causes discomfort, consult a dermatologist for evaluation."
    },
    3: {
        "code": "df",
        "name": "Dermatofibroma",
        "full_name": "Dermatofibroma",
        "risk": "low",
        "description": "A common benign skin nodule, usually found on the legs. It feels like a small, hard bump under the skin and is often brownish in color. May result from minor injuries like insect bites.",
        "recommendation": "Dermatofibromas are benign and typically require no treatment. Surgical removal is an option if the lesion is bothersome or cosmetically undesirable."
    },
    4: {
        "code": "mel",
        "name": "Melanoma",
        "full_name": "Melanoma",
        "risk": "high",
        "description": "The most dangerous form of skin cancer, developing from melanocytes (pigment-producing cells). Often appears as a new dark spot or a change in an existing mole. Follow the ABCDE rule: Asymmetry, Border irregularity, Color variation, Diameter >6mm, Evolution.",
        "recommendation": "URGENT: Seek immediate medical attention. Early detection is critical — melanoma caught early has a 5-year survival rate above 99%. Treatment typically involves surgical excision and may include immunotherapy."
    },
    5: {
        "code": "nv",
        "name": "Melanocytic Nevus",
        "full_name": "Melanocytic Nevus (Common Mole)",
        "risk": "low",
        "description": "A benign proliferation of melanocytes, commonly known as a mole. Can be flat or raised, and vary in color from pink to dark brown. Most adults have 10-40 moles on their body.",
        "recommendation": "Normal moles are harmless. Monitor for any changes using the ABCDE criteria. If a mole changes shape, color, or size, or begins to itch or bleed, schedule a dermatology appointment."
    },
    6: {
        "code": "vasc",
        "name": "Vascular Lesion",
        "full_name": "Vascular Lesions (Angiomas, Angiokeratomas, Pyogenic Granulomas)",
        "risk": "medium",
        "description": "A group of skin lesions arising from blood vessels. Includes cherry angiomas (small red dots), angiokeratomas, pyogenic granulomas, and hemorrhagic lesions.",
        "recommendation": "Most vascular lesions are benign. However, rapidly growing or bleeding lesions should be evaluated by a dermatologist. Treatment options include laser therapy or electrocautery."
    }
}

NUM_CLASSES = 7

# ============================================================
# Image Preprocessing Pipeline
# Must match the validation transforms used during training
# ============================================================

IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

inference_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])

# ============================================================
# Grad-CAM Implementation
# ============================================================

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping for EfficientNet.
    Hooks into the last convolutional layer (model.features[-1])
    to produce a heatmap showing which regions the model focuses on.
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register forward hook to capture activations
        self.fwd_hook = target_layer.register_forward_hook(self._save_activation)
        # Register backward hook to capture gradients
        self.bwd_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, target_class=None):
        """
        Generate a Grad-CAM heatmap for the given input.

        Args:
            input_tensor: Preprocessed image tensor [1, C, H, W]
            target_class: Class index to visualize (None = predicted class)

        Returns:
            cam: Numpy array [H, W] normalized 0-1
        """
        # Forward pass (with gradients enabled)
        self.model.eval()
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Zero existing gradients and backpropagate from target class
        self.model.zero_grad()
        target_score = output[0, target_class]
        target_score.backward()

        # Global average pooling of gradients → channel weights
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)  # [1, C, 1, 1]

        # Weighted combination of activations
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # [1, 1, H, W]
        cam = torch.relu(cam)  # Only positive contributions
        cam = cam.squeeze().cpu().numpy()

        # Normalize to 0-1
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam

    def remove_hooks(self):
        self.fwd_hook.remove()
        self.bwd_hook.remove()


def apply_colormap(heatmap, colormap='jet'):
    """
    Convert a [0,1] grayscale heatmap to an RGB colormap image.
    Uses a pure-numpy JET colormap (no OpenCV dependency).
    """
    # Resize heatmap to IMAGE_SIZE
    heatmap_img = Image.fromarray((heatmap * 255).astype(np.uint8))
    heatmap_img = heatmap_img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    heatmap_resized = np.array(heatmap_img).astype(np.float32) / 255.0

    # JET colormap lookup (pure numpy, matching OpenCV's COLORMAP_JET)
    def jet_colormap(value):
        """Map 0-1 value to RGB using JET colormap."""
        r = np.clip(1.5 - abs(4.0 * value - 3.0), 0, 1)
        g = np.clip(1.5 - abs(4.0 * value - 2.0), 0, 1)
        b = np.clip(1.5 - abs(4.0 * value - 1.0), 0, 1)
        return r, g, b

    h, w = heatmap_resized.shape
    colored = np.zeros((h, w, 3), dtype=np.float32)
    r, g, b = jet_colormap(heatmap_resized)
    colored[:, :, 0] = r
    colored[:, :, 1] = g
    colored[:, :, 2] = b

    return (colored * 255).astype(np.uint8)


def generate_gradcam_images(image_pil, input_tensor, model, device, predicted_idx):
    """
    Generate original image, Grad-CAM overlay, and heatmap as base64 strings.

    Returns:
        dict with 'original', 'gradcam_overlay', 'heatmap' as base64 data URIs
    """
    # Set up Grad-CAM on the last convolutional block
    target_layer = model.features[-1]
    gradcam = GradCAM(model, target_layer)

    try:
        # Need gradients for Grad-CAM, so use a fresh forward pass
        input_with_grad = input_tensor.clone().requires_grad_(True)
        cam = gradcam.generate(input_with_grad, target_class=predicted_idx)
    finally:
        gradcam.remove_hooks()

    # Resize original image
    original_resized = image_pil.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
    original_np = np.array(original_resized)

    # Generate colored heatmap
    heatmap_colored = apply_colormap(cam)

    # Generate overlay (blend original + heatmap)
    alpha = 0.45
    overlay = (original_np.astype(np.float32) * (1 - alpha) +
               heatmap_colored.astype(np.float32) * alpha)
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    # Convert all to base64
    def to_base64(np_img):
        pil_img = Image.fromarray(np_img)
        buffer = io.BytesIO()
        pil_img.save(buffer, format='PNG')
        b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return f"data:image/png;base64,{b64}"

    return {
        "original": to_base64(original_np),
        "gradcam_overlay": to_base64(overlay),
        "heatmap": to_base64(heatmap_colored)
    }


# ============================================================
# Model Loading
# ============================================================

def load_model(model_path, device=None):
    """
    Rebuilds the EfficientNet-B3 architecture and loads the trained weights.

    Args:
        model_path: Path to the .pth state dict file
        device: torch device (auto-detected if None)

    Returns:
        model: Loaded model in eval mode
        device: The device the model is on
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Rebuild the architecture (must match training code exactly)
    model = models.efficientnet_b3(weights=None)  # Don't load ImageNet weights
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, NUM_CLASSES)

    # Load trained weights
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    print(f"[SkinAI] Model loaded successfully on {device}")
    return model, device


# ============================================================
# Prediction
# ============================================================

def predict_image(image_file, model, device):
    """
    Takes a file-like object (uploaded image), preprocesses it,
    runs inference with Grad-CAM, and returns structured results.

    Args:
        image_file: File-like object or path string
        model: Loaded EfficientNet model
        device: torch device

    Returns:
        dict with prediction results and Grad-CAM images
    """
    # Load image
    if isinstance(image_file, str):
        image = Image.open(image_file).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')

    # Preprocess
    input_tensor = inference_transform(image).unsqueeze(0).to(device)

    # Inference (with gradients disabled for fast probabilities)
    with torch.inference_mode():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)

    # Extract results
    confidence, predicted_idx = torch.max(probabilities, 0)
    predicted_idx = predicted_idx.item()
    confidence_val = confidence.item()

    # Generate Grad-CAM visualizations (needs gradients enabled)
    gradcam_input = inference_transform(image).unsqueeze(0).to(device)
    gradcam_images = generate_gradcam_images(
        image, gradcam_input, model, device, predicted_idx
    )

    # Build class-by-class probability breakdown
    all_probabilities = []
    for i in range(NUM_CLASSES):
        info = CLASS_INFO[i]
        all_probabilities.append({
            "code": info["code"],
            "name": info["name"],
            "probability": round(probabilities[i].item() * 100, 2)
        })

    # Sort by probability descending
    all_probabilities.sort(key=lambda x: x["probability"], reverse=True)

    # Build response
    predicted_info = CLASS_INFO[predicted_idx]
    result = {
        "prediction": {
            "code": predicted_info["code"],
            "name": predicted_info["name"],
            "full_name": predicted_info["full_name"],
            "confidence": round(confidence_val * 100, 2),
            "risk": predicted_info["risk"],
            "description": predicted_info["description"],
            "recommendation": predicted_info["recommendation"]
        },
        "all_probabilities": all_probabilities,
        "images": gradcam_images
    }

    return result
