import os
import shutil

# Function to move images from subfolders to the root folder
def move_images_to_root(root_dir):
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png', '.gif')):  # Add other extensions if necessary
                src_path = os.path.join(subdir, file)
                dst_path = os.path.join(root_dir, file)
                shutil.move(src_path, dst_path)
                print(f"Moved {file} to {root_dir}")

# Specify the root directory containing subfolders with images
root_directory = 'images'

# Move images to the root directory
move_images_to_root(root_directory)
