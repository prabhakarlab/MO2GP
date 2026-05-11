# 3. MPEG-7 Dataset 
Next, we used the widely recognized MPEG-7 Dataset. This dataset consists of 70 basic shape categories, with 20 images per category, and is a standard benchmark for evaluating the performance of shape similarity methods.<br>


Reference:<br>
https://dabi.temple.edu/external/shape/MPEG7/dataset.html<br>

The raw imagesof MPEG dataset underwent a preprocessing pipeline to extract the largest contour, following the same method used for our simulation dataset.<br>
In this tutorial, we will use on a few subset of the MPEG-7 dataset, as visualizing all 70 shape categories simultaneously in a UMAP can be challenging to interpret.<br>

3a. MPEG7 dataset 15 shapes
# Load the contour file 
```python
import pickle

# Load contour, label, and image files
with open(r"User_Path\contour_MPEG_15groups.pkl", 'rb') as f:
    contour_input = pickle.load(f)
with open(r"User_Path\label_MPEG_15groups.pkl", 'rb') as f:
    labels = pickle.load(f)
labels = np.array(labels)
img_input = np.load(r"User_Path\image_MPEG_15groups.npy")

# Visualize the processed images 
idx = np.arange(0, labels.shape[0], 20) # start= 0 from the first image,stop=labels.shape[0] = 1400 → go up to 1400 (not inclusive),step=20(pick every 20th image)
fig, ax = plt.subplots(ncols=5, nrows=1, figsize=(20, 4))
ax = ax.flatten()
for i in range(len(idx)):
    temp = img_input[idx[i]]
    ax[i].imshow(temp)
    ax[i].set_title(f"Image {i}")
plt.tight_layout()
plt.show()

# Visualize the contour
idx = np.arange(0, labels.shape[0], 20)
fig, ax = plt.subplots(ncols=5, nrows=1, figsize=(15, 3))
ax = ax.flatten()
for i in range(len(idx)):
    temp = contour_input[idx[i]]
    ax[i].plot(temp[:, 0], temp[:, 1])
    ax[i].invert_yaxis()  # optional, matches image orientation
    ax[i].set_title(f"Contour {i}")
plt.tight_layout()
plt.show()
```
# Run MO2GP analysis 
```python
model_align = ShapeAlign(contours=contour_input)
model_align.preprocess_contours(num_workers=1, n_interp=250, n_smooth=0, scale='perimeter') #
model_align.get_embedding(num_workers=1)

shape_embedding = model_align.shape_embedding
contours = model_align.contours
descriptor = model_align.descriptor

ss = silhouette_score(shape_embedding, labels, metric='euclidean')
print(ss, shape_embedding.shape)
```


