from flask import Flask, request, redirect, url_for, session, jsonify
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import os
import pandas as pd
import torch
from PIL import Image as PILImage
from transformers import CLIPProcessor, CLIPModel, BertTokenizer, BertModel
from tqdm import tqdm

app = Flask(__name__)
app.secret_key = os.urandom(24)

# OAuth 2.0 setup
CLIENT_SECRETS_FILE = r"C:\Users\aks\Desktop\zigguratss\VisualSearch UI\client_secret_776750948178-550t3qgags715h76e6rvea24kd5cdo45.apps.googleusercontent.com.json" # Update this path
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
flow = Flow.from_client_secrets_file(
    CLIENT_SECRETS_FILE,
    scopes=SCOPES,
    redirect_uri='http://localhost:5000/oauth2callback'
)

# Load models and processors
clip_model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
clip_processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# Set up Google Drive API service
def get_drive_service():
    if 'credentials' not in session:
        return redirect(url_for('authorize'))
    
    credentials = flow.credentials
    return build('drive', 'v3', credentials=credentials)

@app.route('/authorize')
def authorize():
    authorization_url, state = flow.authorization_url(
        access_type='offline',
        include_granted_scopes='true'
    )
    session['state'] = state
    return redirect(authorization_url)

@app.route('/oauth2callback')
def oauth2callback():
    flow.fetch_token(authorization_response=request.url)
    credentials = flow.credentials
    session['credentials'] = credentials_to_dict(credentials)
    return redirect(url_for('index'))

def credentials_to_dict(credentials):
    return {
        'token': credentials.token,
        'refresh_token': credentials.refresh_token,
        'token_uri': credentials.token_uri,
        'client_id': credentials.client_id,
        'client_secret': credentials.client_secret,
        'scopes': credentials.scopes
    }

# Function to load and preprocess images from Google Drive
def load_image_from_drive(image_file_id):
    drive_service = get_drive_service()
    try:
        request = drive_service.files().get_media(fileId=image_file_id)
        with open('temp_image.jpg', 'wb') as f:
            downloader = googleapiclient.http.MediaIoBaseDownload(f, request)
            done = False
            while done is False:
                status, done = downloader.next_chunk()
        image = PILImage.open('temp_image.jpg').convert('RGB')
        return clip_processor(images=image, return_tensors="pt")['pixel_values'][0]
    except HttpError as error:
        print(f"An error occurred: {error}")
        return None

# Function to get image file IDs from Google Drive
def get_image_file_ids():
    drive_service = get_drive_service()
    results = drive_service.files().list(
        q="mimeType='image/jpeg' or mimeType='image/png'",
        spaces='drive',
        fields='nextPageToken, files(id, name)'
    ).execute()
    items = results.get('files', [])
    return {item['name']: item['id'] for item in items}

@app.route('/api/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image file part in the request"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        # Save the uploaded file
        image_path = os.path.join('uploads', file.filename)
        file.save(image_path)

        # Process the uploaded image
        uploaded_image = load_image(image_path)

        # Example query text, replace with actual query input if needed
        query_text = request.form.get('query_text', 'Abstract painting with vibrant colors')

        # Perform search and get results
        top_results = search_and_display_artwork(uploaded_image, query_text)
        return jsonify(top_results)

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
