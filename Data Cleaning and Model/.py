import pickle

# Open the pickle file and load the data
with open('embeddings.pkl', 'rb') as f:
    data = pickle.load(f)

# Check if the loaded data is a dictionary or a list
if isinstance(data, dict):
    print("Dictionary keys:", data.keys())
elif isinstance(data, list):
    print("List length:", len(data))
else:
    print("Unknown data format")

# If the data is a list, inspect its structure manually
if isinstance(data, list):
    print("Sample item:", data[0])
