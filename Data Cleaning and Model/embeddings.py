import os
import numpy as np
from numpy.linalg import norm
import pandas as pd
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import pickle
from tqdm import tqdm


model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])


def extract_title(image_folder):
    # Extract title from the folder name
    return os.path.basename(os.path.normpath(image_folder))


def generate_embeddings(image_folder, model):
    normalized_result = []

    title = extract_title(image_folder)

    for filename in os.listdir(image_folder):
        img_path = os.path.join(image_folder, filename)
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        expanded_img_array = np.expand_dims(img_array, axis=0)
        preprocessed_img = preprocess_input(expanded_img_array)
        result = model.predict(preprocessed_img).flatten()
        normalized_result.append((title, result / norm(result)))

    return normalized_result


# Path to the folder containing folders (each folder representing a title)
root_folder = './images/'

# Generate embeddings for images in each folder
all_embeddings = []

pbar = tqdm(os.listdir(root_folder), desc="Processing folders", unit="folder")

for folder_name in pbar:
    # for folder_name in os.listdir(root_folder):
    folder_path = os.path.join(root_folder, folder_name)
    embeddings = generate_embeddings(folder_path, model)
    all_embeddings.extend(embeddings)

# Save embeddings, titles, and corresponding image paths to a pickle file
data = {'embeddings': all_embeddings}
with open('./pickle_files/embeddings.pkl', 'wb') as f:
    pickle.dump(data, f)

# Convert vector data to string format
all_embeddings_str = [(title, [str(val) for val in data]) for title, data in all_embeddings]

# Create DataFrame
df = pd.DataFrame(all_embeddings_str, columns=['Title', 'VectorData'])

# Save DataFrame to CSV
df.to_csv('embeddings.csv', index=False)

print("Embeddings generated and saved successfully.")
