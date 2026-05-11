# Swedish Leaf Dataset
To demonstrate the functionalities of MO2GP, in this tutorial we apply the pipeline on Swedish Leaf dataset from:
Oskar J. O. Söderkvist, “Computer vision classifcation of leaves from swedish trees,” Master’s Thesis, Linkoping University, 2001.

#Images Preprocessing 
MO2GP takes the largest continuous contour for each object in image datasets and converted into a binary images as an input.<br>
First,  extracts the largest external contour and the corresponding processed binary image from each sample and stores the contours, labels, and images for downstream analysis.<br>
```python
from PIL import Image
import numpy as np
import cv2

def get_image_and_contours(img_path, size=128, threshold=128, invert_background=False):
    try:
        # Open image and convert to grayscale
        img = Image.open(img_path).convert("L")  # "L" for grayscale
    except IOError:
        raise FileNotFoundError(f"Cannot load {img_path}")
    
    # Convert to OpenCV format (numpy array)
    img = np.array(img)

    # Normalize the image to the range [0, 255] if it's not already in uint8 format
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Invert the background if it is bright (background needs to be 0)
    if invert_background:
        img = 255 - img

    # Apply binary thresholding to segment the leaf
    _, binary = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)

    # Get original dimensions
    h, w = binary.shape
    # Determine the scaling factor based on the largest dimension
    scale = size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    # Resize while maintaining aspect ratio
    img_resized = cv2.resize(binary, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    # Create a 128x128 black image for padding
    img_padded = np.zeros((size, size), dtype=np.uint8)
    # Calculate padding
    top = (size - new_h) // 2
    bottom = size - new_h - top
    left = (size - new_w) // 2
    right = size - new_w - left
    # Apply padding to the resized image
    img_padded[top:top + new_h, left:left + new_w] = img_resized

    # Find the largest continuous contour (external shape)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    largest_contour = None
    if contours:
        # Select the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        largest_contour = largest_contour.reshape(-1, 2)
    
    return img_padded, largest_contour

def silhouette_score(dataIn, labels, metric='euclidean'):    
    output_sample = silhouette_samples(dataIn, labels, metric=metric)
    unique_labels = np.unique(labels)
    group_means = np.array([output_sample[labels == label].mean(axis=0) for label in unique_labels])
    return np.mean(group_means)
```
```python
from tqdm import tqdm
import os
import pickle
import numpy as np

# Initialize data containers
contour_input = []
labels = []
img_input = []

# Define directory
path = "User_Path\\Leaf_Dataset"

# Get a list of all items in the directory
folders = sorted ([f for f in os.listdir(path) if not f.startswith('.')])

# Initialize counter variable
ino = 0

for cur_folder in tqdm(folders, position=0, leave=True):
    files = [f for f in os.listdir(os.path.join(path, cur_folder)) if not f.startswith('.')]

    labels.extend([cur_folder] * len(files))

    # Loop through each individual file within the current folder
    for file in files:
        img_path = os.path.join(path, cur_folder, file)
        cur_img, cur_cont = get_image_and_contours(
            img_path, 
            size=128, 
            threshold=64, 
            invert_background=False)
        img_input.append(cur_img)
        contour_input.append(cur_cont)

#Finalize and Save Data
img_input = np.array(img_input)

# Open a file in "write binary" (wb) mode to save contour data
with open("User_Path\\Leaf_Dataset_contour\\contour_leaf.pkl", "wb") as f:
    pickle.dump(contour_input, f)

# Open a new file to save the label data
with open("User_Path\\Leaf_Dataset_contour\\label_leaf.pkl", "wb") as f:
    pickle.dump(labels, f)

# Save the img_input NumPy array to a file using NumPy's efficient .npy format
np.save(file="User_Path\\Leaf_Dataset_contour\\image_leaf.npy", arr=img_input)
```
