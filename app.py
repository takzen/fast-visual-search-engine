# app.py
import streamlit as st
from dotenv import load_dotenv
from PIL import Image
import faiss
import json
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
import google.generativeai as genai
import os

# Load environment variables
load_dotenv()

# --- Pre-load all necessary models and data ---
@st.cache_resource
def load_models_and_index():
    """Load FAISS index, image paths, and feature extraction model."""
    # Load FAISS index
    index = faiss.read_index('image_index.faiss')
    
    # Load image paths
    with open('image_paths.json', 'r') as f:
        image_paths = json.load(f)
        
    # Load the ResNet model for embedding the query image
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    model.eval()
    
    # Get the image transformation pipeline
    preprocess = weights.transforms()
    
    return index, image_paths, model, preprocess

# --- Function to get embedding for the query image ---
def get_query_embedding(image, model, preprocess):
    """Converts the user's uploaded image to an embedding."""
    img_t = preprocess(image)
    batch_t = torch.unsqueeze(img_t, 0)
    with torch.no_grad():
        embedding = model(batch_t)
    return embedding.squeeze().numpy().reshape(1, -1).astype('float32')

# --- Function to get Gemini's explanation ---
def get_gemini_explanation(query_image, result_images):
    """Asks Gemini to explain the similarity."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return "Error: GOOGLE_API_KEY not set."
    genai.configure(api_key=api_key)
    
    model = genai.GenerativeModel('gemini-2.5-pro')
    
    prompt = "Explain why these images are visually similar to the first (query) image. Focus on elements, colors, and composition."
    
    # Prepare a list of images for the multimodal prompt
    content = [prompt, query_image]
    content.extend(result_images)
    
    response = model.generate_content(content)
    return response.text

# --- Streamlit App UI ---
st.set_page_config(layout="wide")
st.title("High-Performance Visual Search Engine 🖼️")
st.write("Upload an image to find visually similar ones from a dataset of over 14,000 images, powered by FAISS and ResNet.")

# Load resources
try:
    index, image_paths, model, preprocess = load_models_and_index()
except FileNotFoundError:
    st.error("Index files not found. Please run the `indexer.py` script first to build the index.")
    st.stop()

# --- Image Upload and Search ---
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    query_image = Image.open(uploaded_file).convert("RGB")
    
    st.image(query_image, caption='Your Query Image', width=256)

    if st.button('Find Similar Images'):
        with st.spinner('Searching for similar images...'):
            # 1. Get embedding for the query image
            query_embedding = get_query_embedding(query_image, model, preprocess)
            
            # 2. Search the FAISS index
            k = 5 # Number of similar images to retrieve
            distances, indices = index.search(query_embedding, k)
            
            # 3. Display results
            st.header("Search Results")
            cols = st.columns(k)
            result_images_for_gemini = []
            for i in range(k):
                image_path = image_paths[indices[0][i]]
                image = Image.open(image_path)
                with cols[i]:
                    st.image(image, caption=f"Result {i+1}", use_container_width=True)
                result_images_for_gemini.append(image)

            # 4. (Optional) Get explanation from Gemini
            with st.spinner("Asking Gemini to explain the results..."):
                explanation = get_gemini_explanation(query_image, result_images_for_gemini)
                st.header("AI Explanation of Similarity (from Gemini)")
                st.markdown(explanation)