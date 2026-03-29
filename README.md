# Music Classifier

Music Classifier is a machine learning project for organizing songs into learned clusters based on extracted audio features and model-generated embeddings. It includes training assets, saved models, and a script for classifying new songs into playlist-style groups.

The project is focused on feature extraction, embedding generation, clustering, and automatic song sorting.

## Tech

- Python
- TensorFlow / Keras
- NumPy
- joblib
- Audio feature extraction

## Main Files

- `MODEL.ipynb` contains model experimentation and training work.
- `Features.py` extracts features from audio files.
- `USING_THE_MODEL.py` classifies new songs and moves them into cluster folders.
- `song_embedding_model.keras` and `kmeans_song_cluster.joblib` store the trained models.

