import os
import numpy as np
import pandas as pd
import pickle
import spacy
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

# Path to the directory where you want to save the model
output_dir = 'cache/spacy_model/'

# Check if the model is already downloaded
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    # Download the en_core_web_md model if not found
    spacy.cli.download('en_core_web_md')
    nlp = spacy.load('en_core_web_md')
    nlp.to_disk(output_dir)

# Load the model directly from the project directory
nlp = spacy.load(output_dir)

# Load Myntra Dataset
myntra = pd.read_csv('assets/myntra.csv')


def precompute_title_vectors(titles):
    title_vectors = [nlp(t).vector for t in titles]
    return title_vectors


titles = (myntra['corpusData']).values

if os.path.exists('assets/title_vectors.pkl'):
    # File exists, load title vectors
    title_vectors = pickle.load(open('assets/title_vectors.pkl', 'rb'))
else:
    # File does not exist, compute and save title vectors
    title_vectors = precompute_title_vectors(titles)
    pickle.dump(title_vectors, open('assets/title_vectors.pkl', 'wb'))

# Load pre-trained ResNet50 model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tf.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

# Load the embeddings from the pickle file
with open('assets/embeddings.pkl', 'rb') as f:
    data = pickle.load(f)

vector_data_list = [entry[1] for entry in data['embeddings']]
feature_list = np.array(vector_data_list)

# Fit Nearest Neighbors model with the embeddings
neighbors = NearestNeighbors(n_neighbors=15, algorithm='brute', metric='euclidean')
neighbors.fit(feature_list)


def get_top_matches(search, n=10):
    # Get word vector for the search title
    search_title_vector = nlp(search).vector

    # Calculate cosine similarity between search title vector and title vectors
    similarities = cosine_similarity(search_title_vector.reshape(1, -1), title_vectors)[0]

    # Get indices of top n closest matches
    closest_indices = similarities.argsort()[-n:][::-1]

    # Get top n closest matches and their similarity scores
    closest_matches = [(titles[i], similarities[i]) for i in closest_indices]

    return closest_matches


def get_similar_images(image_data):
    # Load the query image
    image_data = image_data.resize((224, 224))
    img_array = image.img_to_array(image_data)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)

    # Get the feature vector for the query image
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / np.linalg.norm(result)

    # Find nearest neighbors
    indices = neighbors.kneighbors([normalized_result], return_distance=False)

    # print(indices)

    # Get titles of closest results
    closest_titles = [int(data['embeddings'][idx][0]) for idx in indices[0]]

    # print(list(dict.fromkeys(closest_titles)))
    return list(dict.fromkeys(closest_titles))
