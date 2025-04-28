import os
import shutil
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import json

# Load models
embedding_model = load_model('song_embedding_model.keras')
kmeans = joblib.load('kmeans_song_cluster.joblib')

# Your song folder
new_song_folder = 'NEW_SONG_FOLDER'
destination_folder = 'CLUSTERED_SONGS'  # Where you want clustered folders to be created

# Create destination root if it doesn't exist
os.makedirs(destination_folder, exist_ok=True)

# Function to process each song
def process_song(filepath):
    from Features import compile_features
    from tensorflow.keras.preprocessing.image import load_img, img_to_array

    # Extract features
    features, tension, energy, _ = compile_features(filepath, viz=False)
    
    # Load image stack (assuming images are pre-generated in IMAGES/)
    song_name = os.path.splitext(os.path.basename(filepath))[0]
    img_folder = 'IMAGES'  # change if needed
    img_stack = []
    image_names = [
        "chroma_spectrogram.png", "chroma_stripe.png", "mel_spectrogram.png",
        "mfcc_heatmap.png", "mfcc_surface.png", "onset_detection.png",
        "pca_mfcc.png", "tsne_mfcc.png"
    ]
    
    for name in image_names:
        img_path = os.path.join(img_folder, name)
        img = load_img(img_path, target_size=(224, 224))
        img_array = img_to_array(img) / 255.0
        img_stack.append(img_array)

    img_stack = np.concatenate(img_stack, axis=-1)  # (224, 224, 24)

    # Predict embedding
    features = np.expand_dims(features, axis=0)
    img_stack = np.expand_dims(img_stack, axis=0)
    scalars = np.expand_dims([tension, energy], axis=0)
    
    embedding = embedding_model.predict([features, img_stack, scalars])

    return embedding

# Scan the folder
cluster_playlists = {i: [] for i in range(27)} # Assuming 27 clusters

for filename in os.listdir(new_song_folder):
    if filename.endswith('.mp3') or filename.endswith('.wav'):
        song_path = os.path.join(new_song_folder, filename)
        try:
            embedding = process_song(song_path)
            cluster_label = kmeans.predict(embedding)[0]
            cluster_playlists[cluster_label].append(filename)
            print(f"Assigned {filename} to cluster {cluster_label}")

            # Create cluster folder if it doesn't exist
            cluster_folder = os.path.join(destination_folder, f"Cluster_{cluster_label}")
            os.makedirs(cluster_folder, exist_ok=True)

            # Move the song to the cluster folder
            dest_path = os.path.join(cluster_folder, filename)
            shutil.move(song_path, dest_path)

        except Exception as e:
            print(f"Error processing {filename}: {e}")

# Save playlists
with open('playlists.json', 'w') as f:
    json.dump(cluster_playlists, f, indent=4)

print("Playlists created and songs moved into cluster folders!")
