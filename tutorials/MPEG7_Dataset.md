# 3. MPEG-7 Dataset 
Next, we used the widely recognized MPEG-7 Dataset. This dataset consists of 70 basic shape categories, with 20 images per category, and is a standard benchmark for evaluating the performance of shape similarity methods.<br>

Reference:<br>
https://dabi.temple.edu/external/shape/MPEG7/dataset.html<br>

The raw images of MPEG dataset underwent an image preprocessing pipeline to extract the largest contour, following the same method used for Swedish Leaf dataset.The contour,image, and label files are available in `data` folder. 

In this tutorial, we will use a few subset of the MPEG-7 dataset, as visualizing all 70 shape categories simultaneously in a UMAP can be challenging to interpret.<br>
## 3a. MPEG7 dataset 15 shapes
### Load the contour file 
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
idx = np.arange(0, labels.shape[0], 20)
fig, ax = plt.subplots(ncols=5, nrows=4, figsize=(25,20))
ax = ax.flatten()
for i in range(len(idx)):
    temp = img_input[idx[i]]
    ax[i].imshow(temp)
    ax[i].set_title(f"Image {i}")
plt.tight_layout()
plt.show()

# Visualize the contour
idx = np.arange(0, labels.shape[0], 20)
fig, ax = plt.subplots(ncols=5, nrows=4, figsize=(20, 25))
ax = ax.flatten()
for i in range(len(idx)):
    temp = contour_input[idx[i]]
    ax[i].plot(temp[:, 0], temp[:, 1])
    ax[i].invert_yaxis()  # optional, matches image orientation
    ax[i].set_title(f"Contour {i}")
plt.tight_layout()
plt.show()
```
![MPEG7_15shapes_Image](../tutorials/MPEG_results/images_MPEG7_15groups.png)
![MPEG7_15shape_Contour](../tutorials/MPEG_results/contour_MPEG7_15groups.png)
### Run MO2GP analysis 
```python
model_align = ShapeAlign(contours=contour_input)
model_align.preprocess_contours(num_workers=1, n_interp=250, n_smooth=0, scale='perimeter') 
model_align.get_embedding(num_workers=1)

shape_embedding = model_align.shape_embedding
contours = model_align.contours
descriptor = model_align.descriptor

ss = silhouette_score(shape_embedding, labels, metric='euclidean')
print(ss, shape_embedding.shape)
```
### UMAP Visualization
```python
# Define a list of 15 distinct colors 
color_list = [
    (0.788, 0.498, 0.498), # brown
    (0, 0, 0),             # black
    (1.0, 0.647, 0.823),   # hotpink
    (0.701, 0.4, 0.701),   # purple
    (0.4, 0.4, 1.0),       # blue
    (0.4, 0.701, 0.4),     # green
    (0.456, 0.632, 0.779), # steel blue
    (1.0, 0.788, 0.4)     # orange
    (1.0, 0.4, 0.4),       # red
    (0.6, 0.4, 0.2)       # dark brown 
    (0.5, 0.5, 0.5),       # gray
    (0.8, 0.8, 0.0),       # yellow
    (0.5, 0.0, 0.5),       # dark purple
    (0.0, 0.6, 0.6),       # teal
    (1.0, 0.6, 0.0)       # dark orange
]

shapes=['Glas','Heart','bell','brick','cellular_phone',
         'children','device1','device5','flatfish','fork', 
         'fountain','horseshoe','spoon','spring','teddy']

shape_color_dict = dict(zip(shapes, color_list))

fit = umap.UMAP(random_state=19)
embedding = fit.fit_transform(shape_embedding)

for shape in np.unique(labels):
    plt.scatter(
        embedding[labels == shape, 0],
        embedding[labels == shape, 1],
        s=5,
        c=shape_color_dict[shape],
        label=shape
    )

legend_elements = [
    Line2D([0], [0], color=color_list[i], lw=3, label=shapes[i])
    for i in range(5)
]

plt.xlabel('UMAP1')
plt.ylabel('UMAP2')
plt.title(f'Subset of MPEG-7 Dataset UMAP 15 shapes, SI={ss:.4f}', fontweight='bold', fontsize=12)
plt.legend(handles=legend_elements,loc='center left',bbox_to_anchor=(1.02, 0.5), fontsize=15)
plt.show()
```
![MPEG7_15shape_Contour](../tutorials/MPEG_results/MPEG7_MO2GP_UMAP_15groups.png)

### Visualize the representative contour
```python
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Map shape name → index
shape_to_idx = {shape: i for i, shape in enumerate(shapes)}

# Convert string labels → numeric labels
# labels must be a list/array of shape names
species_labels = np.array([shape_to_idx[l] for l in labels])

# UMAP embedding
fit = umap.UMAP(random_state=19)
embedding = fit.fit_transform(shape_embedding)

# pick one representative per species 
representative_indices = []
for species_idx in range(5):
    idxs = np.where(species_labels == species_idx)[0]
    center = embedding[idxs].mean(axis=0)
    dists = np.linalg.norm(embedding[idxs] - center, axis=1)
    representative_indices.append(idxs[np.argmin(dists)])
scale = 1.2
fig, ax = plt.subplots(figsize=(8, 8))

#overlay contours 
for idx in representative_indices:
    contour = contours[idx]
    contour = contour - contour.mean(axis=0)
    # rotate 180° (flip vertically and horizontally)
    theta = np.pi  # 180 degrees
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    contour = contour @ R.T
    # normalize contour size 
    contour = contour / np.max(np.linalg.norm(contour, axis=1))
    # scale
    contour = contour * scale
    # shift to UMAP position
    contour = contour + embedding[idx]
    # add polygon
    ax.add_patch(
        Polygon(
            contour,
            closed=True,
            fill=False,
            edgecolor=color_list[species_labels[idx]],
            linewidth=2.5
        )
    )

legend_elements = [
    Line2D([0], [0], color=color_list[i], lw=3, label=shapes[i])
    for i in range(5)
]

ax.set_xlabel("UMAP1")
ax.set_ylabel("UMAP2")
plt.title(f'Subset of MPEG-7 Dataset UMAP (circle) contour, SI={ss:.4f}',fontweight='bold')
ax.axis("equal")
ax.set_aspect("equal", adjustable="box")
ax.legend(handles=legend_elements,loc='center left',bbox_to_anchor=(1.02, 0.5), fontsize=10)
plt.tight_layout()
plt.show()
```
![MPEG7_15shape_Contour](../tutorials/MPEG_results/MPEG7_MO2GP_UMAP_15groups_contours.png)

## 3b. MPEG7 dataset device groups
This subset of MPEG7 dataset consist of 10 shape groups labelled as device_0 till device_9. The "devices" range from smooth, recognizable geometric shapes to more complex, jagged forms.
### Load the file and visualize 
```python
import pickle

# Load contour, label, and image files
with open(r"User_Path\contour_MPEG_device.pkl", 'rb') as f:
    contour_input = pickle.load(f)
with open(r"User_Path\label_MPEG_device.pkl", 'rb') as f:
    labels = pickle.load(f)
labels = np.array(labels)
img_input = np.load(r"User_Path\image_MPEG_device.npy")

# Visualize the processed images 
idx = np.arange(0, labels.shape[0], 20) # start= 0 from the first image,stop=labels.shape[0] = 1400 → go up to 1400 (not inclusive),step=20(pick every 20th image)
fig, ax = plt.subplots(ncols=5, nrows=2, figsize=(20, 8))
ax = ax.flatten()
for i in range(len(idx)):
    temp = img_input[idx[i]]
    ax[i].imshow(temp)
    ax[i].set_title(f"Image {i}")
plt.tight_layout()
plt.show()

# Visualize the contour
idx = np.arange(0, labels.shape[0], 20)
fig, ax = plt.subplots(ncols=5, nrows=1, figsize=(20, 8))
ax = ax.flatten()
for i in range(len(idx)):
    temp = contour_input[idx[i]]
    ax[i].plot(temp[:, 0], temp[:, 1])
    ax[i].invert_yaxis() 
    ax[i].set_title(f"Contour {i}")
plt.tight_layout()
plt.show()
```
![MPEG7_device_Image](../tutorials/MPEG_results/processed_images_MPEG7_device.png)
![MPEG7_device_Contour](../tutorials/MPEG_results/contour_MPEG7_device.png)

### Run MO2GP analysis 
```python
model_align = ShapeAlign(contours=contour_input)
model_align.preprocess_contours(num_workers=1, n_interp=250, n_smooth=0, scale='perimeter') 
model_align.get_embedding(num_workers=1)

shape_embedding = model_align.shape_embedding
contours = model_align.contours
descriptor = model_align.descriptor

ss = silhouette_score(shape_embedding, labels, metric='euclidean')
print(ss, shape_embedding.shape)
```
### UMAP Visualization
```python
# Define a list of 15 distinct colors 
color_list = [
    (0.788, 0.498, 0.498), # brown
    (0, 0, 0),             # black
    (1.0, 0.647, 0.823),   # hotpink
    (0.701, 0.4, 0.701),   # purple
    (0.4, 0.4, 1.0),       # blue
    (0.4, 0.701, 0.4),     # green
    (0.456, 0.632, 0.779), # steel blue
    (1.0, 0.788, 0.4)     # orange
    (1.0, 0.4, 0.4),       # red
    (0.6, 0.4, 0.2)       # dark brown 
]

shapes=['device0', 'device1', 'device2', 'device3', 'device4','device5', 'device6', 'device7', 'device8', 'device9']

shape_color_dict = dict(zip(shapes, color_list))

fit = umap.UMAP(random_state=19)
embedding = fit.fit_transform(shape_embedding)

for shape in np.unique(labels):
    plt.scatter(
        embedding[labels == shape, 0],
        embedding[labels == shape, 1],
        s=5,
        c=shape_color_dict[shape],
        label=shape
    )

legend_elements = [
    Line2D([0], [0], color=color_list[i], lw=3, label=shapes[i])
    for i in range(10)
]

plt.xlabel('UMAP1')
plt.ylabel('UMAP2')
plt.title(f'Subset of MPEG-7 Dataset UMAP device, SI={ss:.4f}', fontweight='bold', fontsize=12)
plt.legend(handles=legend_elements,loc='center left',bbox_to_anchor=(1.02, 0.5), fontsize=15)
plt.show()
```
![MPEG7_device_UMAP](../tutorials/MPEG_results/MPEG7_MO2GP_UMAP_device.png)
![MPEG7_device_UMAP_contour](../tutorials/MPEG_results/MPEG7_MO2GP_UMAP_device_contour.png)

The UMAP embedding reveals that device4 and device8 are clustered together in the top-right quadrant due to their shared three-pointed or triangular-based geometry. In the bottom-right region, device3, device2, and device5 are grouped together because they all exhibit four-lobed or cross-like structures. Finally, device0 and device7 (rather than device 9) are positioned together in the top-left area because they both possess radial, star-like protrusions.

## 3C. MPEG7 dataset circle groups
This subset of the MPEG-7 dataset consists of five shape categories that share a common circular base geometry: Apple, Device9, HCircle, Octopus, and Pocket.
### Load the file and visualize 
```python
import pickle

# Load files
with open(r"User_Path\contour_MPEG_circle.pkl", 'rb') as f:
    contour_input = pickle.load(f)
with open(r"User_Path\label_MPEG_circle.pkl", 'rb') as f:
    labels = pickle.load(f)
labels = np.array(labels)
img_input = np.load(r"User_Path\image_MPEG_circle.npy")

# Visualize the processed images 
idx = np.arange(0, labels.shape[0], 20) 
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
fig, ax = plt.subplots(ncols=5, nrows=1, figsize=(20, 4))
ax = ax.flatten()
for i in range(len(idx)):
    temp = contour_input[idx[i]]
    ax[i].plot(temp[:, 0], temp[:, 1])
    ax[i].invert_yaxis() 
    ax[i].set_title(f"Contour {i}")
plt.tight_layout()
plt.show()
```
![MPEG7_circle_Image](../tutorials/MPEG_results/processed_images_MPEG7_device.png)
![MPEG7_circle_Contour](../tutorials/MPEG_results/contour_MPEG7_device.png)

### Run MO2GP analysis 
```python
model_align = ShapeAlign(contours=contour_input)
model_align.preprocess_contours(num_workers=1, n_interp=250, n_smooth=0, scale='perimeter') 
model_align.get_embedding(num_workers=1)

shape_embedding = model_align.shape_embedding
contours = model_align.contours
descriptor = model_align.descriptor

ss = silhouette_score(shape_embedding, labels, metric='euclidean')
print(ss, shape_embedding.shape)
```
### UMAP Visualization
```python
# Define a list of 5 distinct colors 
color_list = [
    (0.788, 0.498, 0.498), # brown
    (1.0, 0.647, 0.823),   # hotpink
    (0.701, 0.4, 0.701),   # purple
    (0.4, 0.701, 0.4),     # green
    (1.0, 0.788, 0.4)     # orange
]

shapes=['apple','device9','HCircle','octopus','pocket']

shape_color_dict = dict(zip(shapes, color_list))

fit = umap.UMAP(random_state=19)
embedding = fit.fit_transform(shape_embedding)

for shape in np.unique(labels):
    plt.scatter(
        embedding[labels == shape, 0],
        embedding[labels == shape, 1],
        s=5,
        c=shape_color_dict[shape],
        label=shape
    )

legend_elements = [
    Line2D([0], [0], color=color_list[i], lw=3, label=shapes[i])
    for i in range(5)
]

plt.xlabel('UMAP1')
plt.ylabel('UMAP2')
plt.title(f'Subset of MPEG-7 Dataset UMAP circle, SI={ss:.4f}', fontweight='bold', fontsize=12)
plt.legend(handles=legend_elements,loc='center left',bbox_to_anchor=(1.02, 0.5), fontsize=15)
plt.show()
```
![MPEG7_circle_UMAP](../tutorials/MPEG_results/MPEG7_MO2GP_UMAP_circle.png)
![MPEG7_circle_UMAP_Contour](../tutorials/MPEG_results/MPEG7_MO2GP_UMAP_circle_contour.png)

Despite the circular nature of all five classes, the UMAP showed MO2GP effectively separates the shapes into two distinct zones: the "irregular" shapes (pocket and HCircle) are partitioned toward the left and bottom of the UMAP, while the primary circular cluster (apple, device9, and octopus) occupies the right side of the plot. In this circular group, device9 and octopus are clustered tightly due to their similar high-frequency structural details, whereas the apple is positioned further away because of its low-frequency outline.

## 3D. MPEG7 dataset circle groups
The last subset of MPEG7 dataset is comprised of three groups characterized by their elongated and curving forms: the horseshoe, lizard, and sea_snake.
### Load the file and visualize 
```python
import pickle

# Load files
with open(r"User_Path\contour_MPEG_3_similar_groups.pkl", 'rb') as f:
    contour_input = pickle.load(f)
with open(r"User_Path\label_MPEG_3_similar_groups.pkl", 'rb') as f:
    labels = pickle.load(f)
labels = np.array(labels)
img_input = np.load(r"User_Path\image_MPEG_3_similar_groups.npy")

# Visualize the processed images 
idx = np.arange(0, labels.shape[0], 20) 
fig, ax = plt.subplots(ncols=3, nrows=1, figsize=(15, 4))
ax = ax.flatten()
for i in range(len(idx)):
    temp = img_input[idx[i]]
    ax[i].imshow(temp)
    ax[i].set_title(f"Image {i}")
plt.tight_layout()
plt.show()

# Visualize the contour
idx = np.arange(0, labels.shape[0], 20)
fig, ax = plt.subplots(ncols=3, nrows=1, figsize=(15, 4))
ax = ax.flatten()
for i in range(len(idx)):
    temp = contour_input[idx[i]]
    ax[i].plot(temp[:, 0], temp[:, 1])
    ax[i].invert_yaxis() 
    ax[i].set_title(f"Contour {i}")
plt.tight_layout()
plt.show()
```
![MPEG7_curve_Image](../tutorials/MPEG_results/processed_images_MPEG_3groups_specific_curved.png)
![MPEG7_curve_Contour](../tutorials/MPEG_results/contour_MPEG_3groups_specific_curved.png)

### Run MO2GP analysis 
```python
model_align = ShapeAlign(contours=contour_input)
model_align.preprocess_contours(num_workers=1, n_interp=250, n_smooth=0, scale='perimeter') 
model_align.get_embedding(num_workers=1)

shape_embedding = model_align.shape_embedding
contours = model_align.contours
descriptor = model_align.descriptor

ss = silhouette_score(shape_embedding, labels, metric='euclidean')
print(ss, shape_embedding.shape)
```
### UMAP Visualization
```python
# Define a list of 3 distinct colors 
color_list = [
    (1.0, 0.647, 0.823),   # hotpink
    (0.701, 0.4, 0.701),   # purple
    (0.4, 0.701, 0.4),     # green
]

shapes=['horseshoe','lizzard','sea_snake']

shape_color_dict = dict(zip(shapes, color_list))

fit = umap.UMAP(random_state=19)
embedding = fit.fit_transform(shape_embedding)

for shape in np.unique(labels):
    plt.scatter(
        embedding[labels == shape, 0],
        embedding[labels == shape, 1],
        s=5,
        c=shape_color_dict[shape],
        label=shape
    )

legend_elements = [
    Line2D([0], [0], color=color_list[i], lw=3, label=shapes[i])
    for i in range(3)
]

plt.xlabel('UMAP1')
plt.ylabel('UMAP2')
plt.title(f'Subset of MPEG-7 Dataset UMAP curve, SI={ss:.4f}', fontweight='bold', fontsize=12)
plt.legend(handles=legend_elements,loc='center left',bbox_to_anchor=(1.02, 0.5), fontsize=15)
plt.show()
```
![MPEG7_curve_UMAP](../tutorials/MPEG_results/MPEG7_MO2GP_UMAP_3specificgroups_curved.png)


