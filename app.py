import streamlit as st
import torch
import clip
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load the finetuned model
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("RN50x4", device=DEVICE)
model.load_state_dict(torch.load("fashion_best_model.pt", map_location=DEVICE))
model.eval()  # Set the model to evaluation mode

# Load the saved image embeddings and paths
image_embeddings = np.load("image_embeddings.npy")
with open("image_paths.txt", "r") as f:
    image_paths = f.read().splitlines()

# Streamlit App
st.title("Image Recommendation System")
st.write("Enter a query to find the most relevant images:")

# Text input for the query
text_query = st.text_input("Search Query", "royal watches for men")

if st.button('Search'):
    # Tokenize and encode the query
    text_input = clip.tokenize([text_query]).to(DEVICE)  # Move text_input to the same device as the model
    with torch.no_grad():
        text_embedding = model.encode_text(text_input).cpu().numpy()  # Ensure result is moved to CPU for further processing

    # Compute cosine similarities
    similarities = cosine_similarity(text_embedding, image_embeddings).flatten()

    # Retrieve the top-k images
    top_k = 5
    top_indices = similarities.argsort()[-top_k:][::-1]
    top_images = [image_paths[i] for i in top_indices]
    top_scores = [similarities[i] for i in top_indices]

    # Display results in a single row
    st.subheader(f"Top {top_k} Images for Query: '{text_query}'")
    
    cols = st.columns(top_k)  # Create columns for the top-k images
    for i, img_path in enumerate(top_images):
        img = Image.open(img_path)
        cols[i].image(img, caption=f"Score: {top_scores[i]:.4f}", use_column_width=True, width=100)  # Set width to 100 for smaller resolution
