# MO2GP Project
**Morphology O(2)-invariant General-purpose Projection (MO2GP)** is a shape embedding that is invariant to rotation, reflection, scale, and translation. This pipeline uses initial shape contours, applies a Fourier transform to compute magnitude features, and performs dimensionality reduction to obtain the most informative representations. It is suitable for large-scale morphological datasets. The pipeline is implemented in a Python environment and run using Jupyter Notebook.<br>

## 📥 How to Install

You can install **MO2GP** either directly from GitHub or by cloning the repository for development.

### **Option 1 — Install Directly from GitHub**
Use `pip` to install the latest version from the `main` branch:

```bash
pip install git+https://github.com/Ijoanito/MO2GP.git
```

### **Option 2 — Editable Install (Development Mode)**

```bash
# Clone the repository
git clone https://github.com/Ijoanito/MO2GP.git

# Enter the project directory
cd MO2GP

# Install in editable mode
pip install -e .
```

### **The test after installation**
```bash
from shapealign import ShapeAlign  

import numpy as np

# Example: Create a simple circle contour
circle = np.array([[np.cos(t), np.sin(t)] for t in np.linspace(0, 2*np.pi, 100, endpoint=False)])

shapes = ShapeAlign([circle])
shapes.preprocess_contours()

```
### Example Datasets
To demonstrate the functionalities of MO2GP, in this tutorial we use 1 simulation dataset, 2 well known datasets, and 1 in house spatial transcriptomics data:<br>
**1. Simulation dataset**<br>
**2. Swedish Leaf dataset**<br>
**3. MPEG-7 dataset**<br>
**4. VeraFISH Healthy BMMC dataset**<br>

# 1. Simulation dataset
First, we validated the MO2GP shape embedding using a synthetic dataset of 2,160 simulated cells. This dataset consist of 18 distinct geometric categories derived from nine fundamental shapes—circle, ellipse, triangle, square, clover, pentagon, hexagon, boomerang, and nephroid—each generated at two aspect ratios (1 and 2). We also introduced low-frequency noise to the contours to mimic the real morphological variations found in real biological samples.

### Run MO2GP analysis 
This step is where the MO2GP takes place. MO2GP Shape embedding uses the **ShapeAlign**, which preprocess the raw contours and performs advanced shape analysis using Fourier transforms and dimensionality reduction. The **preprocess_contours** step is a method to standardize all the contours to ensure all the contours are comparable. It processes the raw contours by interpolating, smoothing, and scaling them using the provided parameters, including **num_workers**, **n_interp**, **n_smooth**, and **scale**.<br>
• **n_interp** :The number of points to interpolate for each contour, resulting in contours of uniform size. The default is 250.<br>
• **n_smooth** :The number of smoothing iterations to apply. The default is 0, meaning no smoothing is applied. Smoothing can be useful for some datasets, as it reduces noise and small irregularities and makes the overall shape easier to interpret.<br>
• **scale**    :The method for scaling the contours to make them size-invariant. Currently only can use perimeter or area. The default is perimeter.<br>
• **num_workers**:The number of parallel workers to use for processing. This speeds up preprocessing when many contours are present. The default is 1; if 1 is chosen, processing is done sequentially.<br>

Next, **get_embedding** is used to compute shape embeddings from the preprocessed contours using a Fourier-based method, which involves : 
1. Converting representative contours coordinate to complex number<br>
2.Applying the Fast Fourier Transform to the contour coordinates breaks them into a combination of waves at different frequencies and representing the boundary as a sum of sinusoidal waves<br>
3. Scalling Fourier coefficients to emphasize certain frequencies and making the features more robust (a form of normalization to improve PCA results)<br>
4. Feature Selection (keep the most informative features)<br>
5. Applying PCA for dimensionality reduction<br>

Finally, the quality of the embeddings is evaluated using a silhouette score to assess how well the features separate the shape classes.

In this tutorial, we set **n_interp** to 250 points, **scale** the shapes by perimeter so that all shapes have the same perimeter length, and used **num_smooth = 0**.<br>

```python
# Load the contour file 
with open("User_Path\\contour_simulation_list_18groups_2160.pkl", "rb") as f:
    contour_input = pickle.load(f)
with open("User_Path\\label_simulation_list_18groups_2160.pkl","rb") as f:
    labels = pickle.load(f)
labels = np.array(labels)

# Start shape analysis 
model_align = ShapeAlign(contours=contour_input)
model_align.preprocess_contours(num_workers=1, n_interp=250, n_smooth=0, scale='area')
model_align.get_embedding(num_workers=1)

shape_embedding = model_align.shape_embedding
contours = model_align.contours
descriptor = model_align.descriptor

ss = silhouette_score(shape_embedding, labels, metric='euclidean')
print(ss, shape_embedding.shape)
```
### Visualize the UMAP
```python
color_list = [
    (0.788, 0.498, 0.498), # brown
    (0, 0, 0),             # black
    (1.0, 0.647, 0.823),   # hotpink
    (0.701, 0.4, 0.701),   # purple
    (0.4, 0.4, 1.0),       # blue
    (0.4, 0.701, 0.4),     # green
    (0.456, 0.632, 0.779), # steel blue
    (1.0, 0.788, 0.4),     # orange
    (1.0, 0.4, 0.4),       # red
    (0.6, 0.4, 0.2),       # dark brown
    (0.5, 0.5, 0.5),       # gray
    (0.8, 0.8, 0.0),       # yellow
    (0.5, 0.0, 0.5),       # dark purple
    (0.0, 0.6, 0.6),       # teal
    (1.0, 0.6, 0.0),       # dark orange
    (0.489, 1.0, 0.0),     # lime green
    (0.5, 0.0, 0.0),       # maroon
    (0.824, 0.706, 0.549), # tan
    (0.25, 0.88, 0.82),    # turquoise
    (0.5, 0.5, 0.0)        # olive
]

shapes = [
    'boomerang', 
    'boomerang aspect', 
    'circle', 
    'circle aspect', 
    'clover', 
    'clover aspect', 
    'hexagon', 
    'hexagon aspect', 
    'nephroid', 
    'nephroid aspect', 
    'pentagon', 
    'pentagon aspect', 
    'square', 
    'square aspect', 
    'star', 
    'star aspect', 
    'triangle', 
    'triangle aspect'
]

shape_color_dict = dict(zip(shapes, color_list))
shape_color_dict

fit = umap.UMAP(n_neighbors=50, min_dist=0.2, random_state=18)

embedding = fit.fit_transform(shape_embedding)

plt.figure(figsize=(12,8))
for shape in np.unique(labels):
    plt.scatter(
        embedding[labels == shape, 0],
        embedding[labels == shape, 1],
        s=5, c=[shape_color_dict[shape]],
        label=shape
    )
plt.axis('equal')
plt.xlabel('UMAP1', fontsize=14)
plt.ylabel('UMAP2', fontsize=14)
plt.tick_params(axis='both', which='major', labelsize=12)
plt.legend(title='Shapes', loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8, title_fontsize=14)
plt.title(f'MO2GP Simulation Data, SI={ss:.4f}',fontsize=16)
plt.show()
```



[Swedish Leaf Dataset](./tutorials/Swedish_Leaf_Dataset.md)

[MPEG7 Dataset](./tutorials/MPEG7_Dataset.md) 

[VeraFISH_Healthy_BMMC_Dataset](./tutorials/VeraFISH_Healthy_BMMC_Dataset.md) 















## Running ShapeAlign in Jupyter Notebook

Below is an example of how to load the pre-processed "leaf" data (`contour_leaf.pkl`, `label_leaf.pkl`, `image_leaf.npy`),  
plot sample images and contours, run `ShapeAlign` to generate shape embeddings, and then visualize the result using UMAP.

```python
import pickle
import numpy as np
import time
import matplotlib.pyplot as plt
import umap
from align import ShapeAlign

# Path to your files
path = "your_path_here"

# Load contour, label, and image files
with open(path + 'contour_leaf.pkl', 'rb') as f:
    contour_input = pickle.load(f)
with open(path + 'label_leaf.pkl', 'rb') as f:
    labels = pickle.load(f)
labels = np.array(labels)
img_input = np.load(path + 'image_leaf.npy')

# --- Plot images ---
idx = np.arange(5, labels.shape[0], 75)
fig, ax = plt.subplots(ncols=5, nrows=3, figsize=(25, 15))
ax = ax.flatten()
for i in range(len(idx)):
    temp = img_input[idx[i]]
    ax[i].imshow(temp)
    ax[i].set_title(f"Image {i}")
plt.show()

# --- Plot contours ---
idx = np.arange(4, labels.shape[0], 75)
fig, ax = plt.subplots(ncols=5, nrows=3, figsize=(25, 15))
ax = ax.flatten()
for i in range(len(idx)):
    temp = contour_input[idx[i]]
    ax[i].plot(temp[:, 0], temp[:, 1])
    ax[i].invert_yaxis()  # optional, matches image orientation
    ax[i].set_title(f"Contour {i}")
plt.show()

# Start timing
start = time.time()

# Initialize ShapeAlign model
model_align = ShapeAlign(contours=contour_input)

# Preprocess contours - centering, interpolation, smoothing
model_align.preprocess_contours(
    num_workers=8,
    n_interp=250,
    n_smooth=0,
    scale='perimeter'
)

# Compute functions of the contour
model_align.get_embedding(num_workers=1)
print('Running time: ', time.time() - start)

# Retrieve results
shape_embedding = model_align.shape_embedding
contours = model_align.contours

# Compute silhouette score
ss = my_silhouette_score(shape_embedding, labels, metric='euclidean')
print(ss, shape_embedding.shape)

# --- Creating the color dictionary for UMAP plot ---
color_list = [
    (0.788, 0.498, 0.498),     # brown
    (0, 0, 0),                 # black
    (1.0, 0.647, 0.823),       # hotpink
    (0.701, 0.4, 0.701),       # purple
    (0.4, 0.4, 1.0),           # blue
    (0.4, 0.701, 0.4),         # green
    (0.456, 0.632, 0.779),     # steel blue
    (1.0, 0.788, 0.4),         # orange
    (1.0, 0.4, 0.4),           # red
    (0.6, 0.4, 0.2),           # dark brown
    (0.5, 0.5, 0.5),           # gray
    (0.8, 0.8, 0.0),           # yellow
    (0.5, 0.0, 0.5),           # dark purple
    (0.0, 0.6, 0.6),           # teal
    (1.0, 0.6, 0.0)            # dark orange
]
shapes = [
    '01.Ulmus_carpinifolia', '02.Acer', '03.Salix_aurita',
    '04.Quercus', '05.Alnus_incana', '06.Betula_pubescens',
    '07.Salix_alba_Sericea', '08.Populus_tremula', '09.Ulmus_glabra',
    '10.Sorbus_aucuparia', '11.Salix_sinerea', '12.Populus',
    '13.Tilia', '14.Sorbus intermedia', '15.Fagus silvatica'
]
shape_color_dict = dict(zip(shapes, color_list))

# --- Run UMAP ---
fit = umap.UMAP(random_state=19)
embedding = fit.fit_transform(shape_embedding)

# Plot UMAP result
for shape in np.unique(labels):
    plt.scatter(
        embedding[labels == shape, 0],
        embedding[labels == shape, 1],
        s=5, c=[shape_color_dict[shape]],
        label=shape
    )
plt.axis('equal')
plt.xlabel('UMAP1')
plt.ylabel('UMAP2')
plt.legend(title='', loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
plt.show()
```

## Example Leaf Plots

### Leaf Images
![Sample Images](example/leaf_results/leaf_image.png)

### Leaf Contours
![Sample Contours](example/leaf_results/leaf_contour.png)

### Leaf UMAP Visualization after MO2GP
![UMAP Result](example/leaf_results/leaf_UMAP.png)

### Leaf UMAP Visualization after MO2GP (Contours on UMAP)
![UMAP Contour Result](example/leaf_results/leaf_UMAP_with_contours_inside.png)
