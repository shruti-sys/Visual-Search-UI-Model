# import streamlit as st
# import torch
# import pickle
# import pandas as pd
# from PIL import Image
# import os
# from transformers import CLIPProcessor, CLIPModel, BertTokenizer, BertModel
# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np

# # Load pre-computed embeddings
# with open(r"C:\Users\shrut\OneDrive\Desktop\ZIGGURATTS\image_embeddings.pkl", 'rb') as f:
#     image_embeddings = pickle.load(f)

# with open(r"C:\Users\shrut\OneDrive\Desktop\ZIGGURATTS\text_embeddings.pkl", 'rb') as f:
#     text_embeddings = pickle.load(f)


# # Load CSV file
# df = pd.read_csv(r'C:\Users\shrut\OneDrive\Desktop\ZIGGURATTS\modified_csv_file2.csv')

# # Load models
# clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
# clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# bert_model = BertModel.from_pretrained('bert-base-uncased')

# # Function to get text embeddings
# def get_text_embedding(text):
#     inputs = bert_tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
#     with torch.no_grad():
#         outputs = bert_model(**inputs)
#     return outputs.last_hidden_state.mean(dim=1).numpy()

# # Function to get image embeddings
# def get_image_embedding(image):
#     inputs = clip_processor(images=image, return_tensors="pt")
#     with torch.no_grad():
#         image_features = clip_model.get_image_features(**inputs)
#     return image_features.numpy()

# # Function to find similar items
# def find_similar_items(query_embedding, embeddings, top_k=10):
#     similarities = cosine_similarity(query_embedding, embeddings)
#     top_indices = np.argsort(similarities[0])[::-1][:top_k]
#     return top_indices

# # Streamlit UI
# st.title("Art Search Engine")

# search_type = st.radio("Choose search type:", ("Text", "Image"))

# if search_type == "Text":
#     query = st.text_input("Enter your search query:")
#     if query:
#         query_embedding = get_text_embedding(query)
#         similar_indices = find_similar_items(query_embedding, text_embeddings)
        
#         st.subheader("Top 10 Results:")
#         for idx in similar_indices:
#             st.write(f"Description: {df['Description'].iloc[idx]}")
#             st.write(f"Keywords: {df['Keywords'].iloc[idx]}")
#             image_path = os.path.join(r"C:\Users\shrut\OneDrive\Desktop\ZIGGURATTS\artwork-dataset", df['Image'].iloc[idx])
#             st.image(image_path, caption=df['Image'].iloc[idx], use_column_width=True)
#             st.write("---")

# else:
#     uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
#     if uploaded_file is not None:
#         image = Image.open(uploaded_file)
#         st.image(image, caption="Uploaded Image", use_column_width=True)
        
#         query_embedding = get_image_embedding(image)
#         similar_indices = find_similar_items(query_embedding, image_embeddings)
        
#         st.subheader("Top 10 Results:")
#         for idx in similar_indices:
#             st.write(f"Description: {df['Description'].iloc[idx]}")
#             st.write(f"Keywords: {df['Keywords'].iloc[idx]}")
#             image_path = os.path.join(r"C:\Users\shrut\OneDrive\Desktop\ZIGGURATTS\artwork-dataset", df['Image'].iloc[idx])
#             st.image(image_path, caption=df['Image'].iloc[idx], use_column_width=True)
#             st.write("---")

# --------------
import streamlit as st
import torch
import pickle
import pandas as pd
from PIL import Image
import os
from transformers import CLIPProcessor, CLIPModel, BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load pre-computed embeddings
with open(r"C:\Users\shrut\OneDrive\Desktop\ZIGGURATTS\image_embeddings.pkl", 'rb') as f:
    image_embeddings = pickle.load(f)

with open(r"C:\Users\shrut\OneDrive\Desktop\ZIGGURATTS\text_embeddings.pkl", 'rb') as f:
    text_embeddings = pickle.load(f)

# Load CSV file
df = pd.read_csv(r'C:\Users\shrut\OneDrive\Desktop\ZIGGURATTS\modified_csv_file2.csv')

# Load models
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# Function to get text embeddings
def get_text_embedding(text):
    inputs = bert_tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy()

# Function to get image embeddings
def get_image_embedding(image):
    inputs = clip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        image_features = clip_model.get_image_features(**inputs)
    return image_features.numpy()

# Function to find similar items
def find_similar_items(query_embedding, embeddings, top_k=10):
    # Reshape the embeddings to ensure they are 2D
    query_embedding = query_embedding.squeeze().reshape(1, -1)
    embeddings = embeddings.squeeze().reshape(embeddings.shape[0], -1)

    # Calculate cosine similarity
    similarities = cosine_similarity(query_embedding, embeddings)
    top_indices = np.argsort(similarities[0])[::-1][:top_k]
    return top_indices

# Streamlit UI
st.title("Art Search Engine")

search_type = st.radio("Choose search type:", ("Text", "Image"))

if isinstance(image_embeddings, dict):
    image_embeddings = np.array([v for v in image_embeddings.values()])

if isinstance(text_embeddings, dict):
    text_embeddings = np.array([v for v in text_embeddings.values()])

if search_type == "Text":
    query = st.text_input("Enter your search query:")
    if query:
        query_embedding = get_text_embedding(query)
        similar_indices = find_similar_items(query_embedding, text_embeddings)
        
        st.subheader("Top 10 Results:")
        for i, idx in enumerate(similar_indices):
            if i % 3 == 0:
                cols = st.columns(3)  # Create 3 columns
            
            with cols[i % 3]:
                st.write(f"Description: {df['Description'].iloc[idx]}")
                st.write(f"Keywords: {df['Keywords'].iloc[idx]}")
                image_path = os.path.join(r"C:\Users\shrut\OneDrive\Desktop\ZIGGURATTS\artwork-dataset", df['Image'].iloc[idx])
                st.image(image_path, caption=df['Image'].iloc[idx], use_column_width=True)

else:
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        query_embedding = get_image_embedding(image)
        similar_indices = find_similar_items(query_embedding, image_embeddings)
        
        st.subheader("Top 10 Results:")
        for i, idx in enumerate(similar_indices):
            if i % 3 == 0:
                cols = st.columns(3)  # Create 3 columns
            
            with cols[i % 3]:
                st.write(f"Description: {df['Description'].iloc[idx]}")
                st.write(f"Keywords: {df['Keywords'].iloc[idx]}")
                image_path = os.path.join(r"C:\Users\shrut\OneDrive\Desktop\ZIGGURATTS\artwork-dataset", df['Image'].iloc[idx])
                st.image(image_path, caption=df['Image'].iloc[idx], use_column_width=True)
#streamlit run c:\Users\shrut\OneDrive\Desktop\ZIGGURATTS\final_app.py [ARGUMENTS]