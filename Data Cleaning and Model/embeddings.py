import os
import numpy as np
import cv2
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
import pickle

def load_image(img_path):
    try:
        return cv2.imread(img_path)
    except Exception as e:
        print("An error occurred while loading the image:", e)
        return None

def preprocess_image(img):
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = preprocess_input(img)
    return img

def generate_embeddings(image_folder, model):
    embeddings = []
    img_paths = []

    for filename in os.listdir(image_folder):
        img_path = os.path.join(image_folder, filename)
        img = load_image(img_path)
        if img is not None:
            img = preprocess_image(img)
            img = np.expand_dims(img, axis=0)
            emb = model.predict(img).flatten()
            embeddings.append(emb)
            img_paths.append(img_path)

    return embeddings, img_paths

# Load pre-trained ResNet50 model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Path to the folder containing images
image_folder = './images/'

# Generate embeddings for images in the folder
embeddings, img_paths = generate_embeddings(image_folder, model)

# Save embeddings and corresponding image paths to a pickle file
data = {'embeddings': embeddings, 'img_paths': img_paths}
with open('embeddings.pkl', 'wb') as f:
    pickle.dump(data, f)

print("Embeddings generated and saved successfully.")
