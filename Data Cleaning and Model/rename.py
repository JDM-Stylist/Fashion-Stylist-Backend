import os

# Directory containing the images
image_dir = "./images"

# List all files in the directory
image_files = os.listdir(image_dir)

# Sort the files alphabetically
image_files.sort()

# Rename the files with indices
for i, filename in enumerate(image_files):
    # Construct the new filename with index
    new_filename = f"{i}.jpg"
    # Construct the full path to the old and new files
    old_filepath = os.path.join(image_dir, filename)
    new_filepath = os.path.join(image_dir, new_filename)
    # Rename the file
    os.rename(old_filepath, new_filepath)
