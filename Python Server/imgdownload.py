import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import pickle

def load_image(img_path):
    try:
        return cv2.imread(img_path)
    except Exception as e:
        print("An error occurred while loading the image:", e)
        return None

def get_recommender(embeddings, input_embedding, top_n=16):
    cosine_sim = cosine_similarity(embeddings, input_embedding.reshape(1, -1))
    sim_scores = list(enumerate(cosine_sim.flatten()))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[:top_n]
    idx_rec = [i[0] for i in sim_scores]
    return idx_rec

# Load only the embeddings column from embeddings.pkl
with open('embeddings.pkl', 'rb') as f:
    embeddings = pickle.load(f)['embeddings']

# Get user input image index
input_image_index = int(input("Enter the index of the input image: "))
if input_image_index < 0 or input_image_index >= len(embeddings):
    print("Invalid input image index.")
    exit()

input_embedding = embeddings[input_image_index]

# Generate recommendations
idx_rec = get_recommender(embeddings, input_embedding)

# Display recommended images
rows = 4
cols = 4
fig, axes = plt.subplots(rows, cols, figsize=(12, 12))
for idx, img_idx in enumerate(idx_rec):
    row = idx // cols
    col = idx % cols
    img_path = f"./images/{img_idx}.jpg"  # Assuming images are named with indices
    img = load_image(img_path)
    axes[row, col].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[row, col].set_title(f"Recommendation {idx+1}")
    axes[row, col].axis('off')

plt.tight_layout()
plt.show()
