
import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import sys
import os
import timm

# Add project root to path to import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import ModifiedModel

CLASSES = [
    "banded", "blotchy", "braided", "bubbly", "bumpy", "chequered",
    "cobwebbed", "cracked", "crosshatched", "crystalline", "dotted",
    "fibrous", "flecked", "freckled", "frilly", "gauzy", "grid",
    "grooved", "honeycombed", "interlaced", "knitted", "lacelike",
    "lined", "marbled", "matted", "meshed", "paisley", "perforated",
    "pitted", "pleated", "polka-dotted", "porous", "potholed",
    "scaly", "smeared", "spiraled", "sprinkled", "stained",
    "stratified", "striped", "studded", "swirly", "veined",
    "waffled", "woven", "wrinkled", "zigzagged"
]

@st.cache_resource
def load_model():
    """
    Loads the trained MoE-LoRA model.
    """
    # Model Configuration (matching config/deit_moe_lora.yaml)
    model_config = {
        'mode': 'moe_lora',
        'base_model_name': 'deit_small_patch16_224',
        'num_experts': 4,
        'lora_rank': 7,
        'lora_alpha': 14,
        'k': 2,
        'target_blocks_indices': [8, 9, 10, 11]
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize Base Model
    # Note: pretrained=True needed to load the base structure correctly before loading our weights? 
    # Usually safer to load structure matching training. 
    # Assuming 'deit_small_patch16_224' is available via timm.
    base_model = timm.create_model(model_config['base_model_name'], pretrained=True)
    
    # Initialize Modified Model
    model = ModifiedModel(
        base_model=base_model,
        num_classes=len(CLASSES),
        model_config=model_config
    )
    
    # Load Weights
    checkpoint_path = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'models', 'deit_moe_lora', 'best_deit_moe_lora_model.pth')
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
            
        model.load_state_dict(state_dict)
    else:
        st.error(f"Checkpoint not found at: {checkpoint_path}")
        return None

    model.to(device)
    model.eval()
    return model

def predict(image, model):
    """
    Predicts the texture class of an image.
    
    Args:
        image (PIL.Image): Input image.
        model (nn.Module): Loaded model.
        
    Returns:
        tuple: (Class Name, Confidence Score)
    """
    device = next(model.parameters()).device
    
    # Strict Preprocessing matches training:
    # 1. Resize to 256x256
    # 2. Center Crop to 224x224
    # 3. Normalize using ImageNet statistics
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits, _ = model(img_tensor)
        probs = torch.softmax(logits, dim=1)
        confidence, predicted_idx = torch.max(probs, 1)
        
    class_name = CLASSES[predicted_idx.item()]
    score = confidence.item() * 100
    
    return class_name, score
