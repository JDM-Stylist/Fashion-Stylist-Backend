import spacy
import os
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity

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
