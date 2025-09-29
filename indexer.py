# indexer.py
import os
import json
import faiss
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights

print("--- Starting Image Indexing Process ---")

# --- 1. Setup Device (CUDA or CPU) ---
# Check if CUDA is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 2. Load Pre-trained Model for Feature Extraction ---
print("Loading ResNet-18 model...")
weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)
model = torch.nn.Sequential(*(list(model.children())[:-1]))

# --- KEY CHANGE: Move the model to the selected device ---
model.to(device)
model.eval()

preprocess = weights.transforms()
print("Model loaded and moved to device.")

# --- 3. Function to Extract Features from an Image ---
def get_image_embedding(image_path, model, preprocess, device):
    """Converts an image to a feature vector (embedding) on the specified device."""
    try:
        img = Image.open(image_path).convert("RGB")
        img_t = preprocess(img)
        # Add a batch dimension and move the tensor to the device
        batch_t = torch.unsqueeze(img_t, 0).to(device)
        
        with torch.no_grad():
            embedding = model(batch_t)
            
        # Move the embedding back to the CPU for NumPy conversion
        return embedding.squeeze().cpu().numpy()
    except Exception as e:
        print(f"Skipping file {image_path}: {e}")
        return None

# --- 4. Process All Images in the Dataset ---
dataset_path = 'data/seg_train'
image_paths = []
for root, _, files in os.walk(dataset_path):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_paths.append(os.path.join(root, file))

print(f"Found {len(image_paths)} images to process.")

all_embeddings = []
valid_paths = []
for i, path in enumerate(image_paths):
    # Pass the device to the function
    embedding = get_image_embedding(path, model, preprocess, device)
    if embedding is not None:
        all_embeddings.append(embedding)
        valid_paths.append(path)
    if (i + 1) % 100 == 0:
        print(f"Processed {i + 1}/{len(image_paths)} images...")

embeddings_np = np.array(all_embeddings).astype('float32')
print("All images converted to embeddings.")

# --- 5. Build and Save the FAISS Index ---
print("Building FAISS index...")
dimension = embeddings_np.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings_np)
print(f"FAISS index built. Total vectors in index: {index.ntotal}")

faiss.write_index(index, 'image_index.faiss')
with open('image_paths.json', 'w') as f:
    json.dump(valid_paths, f)
    
print("--- Indexing complete. ---")