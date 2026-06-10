# 2.Swedish Leaf Dataset
We also applied MO2GP to the Swedish Leaf Dataset, which consists of 15 distinct leaf species with 75 examples each, totaling 1,125 images. <br>

Reference:<br>
Söderkvist, O. J. O. (2001). Computer vision classification of leaves from Swedish trees (Master's Thesis). Linköping University.

## Images Preprocessing <br>
Since, MO2GP utilizes continuous contour from each object as the primary input, we need to preprocess workflow the original leaf images. Here, we extracted the largest external contour and generates a corresponding binary mask for each sample using opencv pipeline. <br>

These contours, along with their associated labels and processed images, have been pre-packaged for downstream shape embedding and are automatically available through the MO2GP `datasets` module. <br>

## Load the pre-processed file and visualize the image and contour file.<br>
```python
import numpy as np
import matplotlib.pyplot as plt
from mo2gp import datasets

# 1. Load the Swedish Leaf data
leaf_data = datasets.load_leaf()
contour_input = leaf_data["contour"]
img_input = leaf_data["images"]
labels = leaf_data["labels"]

# Visualize the processed images
idx = np.arange(5, labels.shape[0], 75)
fig, ax = plt.subplots(ncols=5, nrows=3, figsize=(25, 15))
ax = ax.flatten()
for i in range(len(idx)):
    temp = img_input[idx[i]]
    ax[i].imshow(temp)
    ax[i].set_title(f"Image {i}")
plt.show()

# Visualize the contour
fig, ax = plt.subplots(ncols=5, nrows=3, figsize=(25, 15))
ax = ax.flatten()
for i in range(len(idx)):
    temp = contour_input[idx[i]]
    ax[i].plot(temp[:, 0], temp[:, 1])
    ax[i].invert_yaxis()  # optional, matches image orientation
    ax[i].set_title(f"Contour {i}")
plt.show()
```
![Leaf_Image](../tutorials/Swedish_Leaf_data_results/leaf_image.png)
![Leaf_Contour](../tutorials/Swedish_Leaf_data_results/leaf_contour.png)
**Note:** In the source data, **11.Salix_sineria** is identical to **11.Salix_cineacea** in the manuscript. The original dataset contains a typo.


## Run MO2GP shape embedding 
Now we process the leaf contours through the MO2GP pipeline and evaluate the resulting embeddings.

```python
import numpy as np
from sklearn.metrics import silhouette_samples
from mo2gp import ShapeAlign

# 1. Define the custom group-averaged silhouette score
def silhouette_score(dataIn, labels, metric='euclidean'):    
    output_sample = silhouette_samples(dataIn, labels, metric=metric)
    unique_labels = np.unique(labels)
    group_means = np.array([output_sample[labels == label].mean(axis=0) for label in unique_labels])
    return np.mean(group_means)

# 2. Initialize and run MO2GP
model_align = ShapeAlign(contours=contour_input)
model_align.preprocess_contours(num_workers=1)
model_align.get_embedding()

# 3. Extract the results
shape_embedding = model_align.shape_embedding
contours = model_align.contours
descriptor = model_align.descriptor

# 4. Evaluate embedding quality
ss = silhouette_score(shape_embedding, labels, metric='euclidean')
print(f"Silhouette Score = {ss:.4f}, Embedding Shape: {shape_embedding.shape}")
```

## Generate UMAP for MO2GP visualization

To evaluate how well MO2GP separates the different leaf species, we project the high-dimensional shape embeddings down to 2D using UMAP.
```python
import umap
import matplotlib.pyplot as plt

# Define a list of 15 distinct RGB colors 
color_list = [
    (0.788, 0.498, 0.498), # brown
    (0.0, 0.0, 0.0),       # black
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
    (1.0, 0.6, 0.0)        # dark orange
]

# Define the list of 15 leaf class names
shapes = [
    '01.Ulmus_carpinifolia', '02.Acer', '03.Salix_aurita',
    '04.Quercus', '05.Alnus_incana', '06.Betula_pubescens',
    '07.Salix_alba_Sericea', '08.Populus_tremula', '09.Ulmus_glabra',
    '10.Sorbus_aucuparia', '11.Salix_sinerea', '12.Populus',
    '13.Tilia', '14.Sorbus intermedia', '15.Fagus silvatica'
]

# Create a dictionary mapping each species name to a specific color
shape_color_dict = dict(zip(shapes, color_list))

# Run UMAP dimensionality reduction
fit = umap.UMAP(random_state=18)
embedding = fit.fit_transform(shape_embedding)

# Plot the UMAP results
fig, ax = plt.subplots(figsize=(12, 9))
for shape in np.unique(labels):
    ax.scatter(
        embedding[labels == shape, 0],
        embedding[labels == shape, 1],
        s=5,
        c=[shape_color_dict[shape]],  # Wrapped in brackets to prevent RGB dimension errors
        label=shape
    )

ax.axis('equal')
ax.set_xlabel('UMAP1', fontsize=12)
ax.set_ylabel('UMAP2', fontsize=12)
ax.set_title(f'Swedish Leaf Dataset, SI={ss:.4f}', fontsize=24)

# Position the legend outside the plot
ax.legend(title='Leaf Species', loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12)

fig.tight_layout()
plt.show()
```
![Leaf_UMAP](../tutorials/Swedish_Leaf_data_results/Leafdata_UMAP.png)

User also can visualize the shape represented by each cluster using the code below :
``` python
import umap
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.lines import Line2D

# Define a list of 15 distinct RGB colors 
color_list = [
    (0.788, 0.498, 0.498), # brown
    (0.0, 0.0, 0.0),       # black
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
    (1.0, 0.6, 0.0)        # dark orange
]

# Define the list of 15 leaf class names (matching data labels exactly)
shapes = [
    '01.Ulmus_carpinifolia', '02.Acer', '03.Salix_aurita',
    '04.Quercus', '05.Alnus_incana', '06.Betula_pubescens',
    '07.Salix_alba_Sericea', '08.Populus_tremula', '09.Ulmus_glabra',
    '10.Sorbus_aucuparia', '11.Salix_sinerea', '12.Populus',
    '13.Tilia', '14.Sorbus intermedia', '15.Fagus silvatica'
]

# Create a dictionary mapping each species name to a specific color
shape_color_dict = dict(zip(shapes, color_list))

# Run UMAP dimensionality reduction
fit = umap.UMAP(n_neighbors=50, min_dist=0.2, random_state=18)
embedding = fit.fit_transform(shape_embedding)

# Find the representative index (closest to the centroid) for each species
representative_indices = []
for shape in shapes:
    idxs = np.where(labels == shape)[0]
    if len(idxs) > 0:
        center = embedding[idxs].mean(axis=0)
        dists = np.linalg.norm(embedding[idxs] - center, axis=1)
        representative_indices.append(idxs[np.argmin(dists)])

# Plot the representative contours
scale = 3
fig, ax = plt.subplots(figsize=(12, 9))

for idx in representative_indices:
    shape_name = labels[idx]
    contour = contours[idx].copy()
    contour = contour - contour.mean(axis=0)
    
    # Rotate 180° (flip vertically and horizontally)
    theta = np.pi 
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    contour = contour @ R.T
    
    # Normalize contour size and scale it
    contour = contour / np.max(np.linalg.norm(contour, axis=1)) 
    contour = contour * scale
    
    # Shift the contour to its actual UMAP coordinate position
    contour = contour + embedding[idx]

    # Add the polygon to the plot
    ax.add_patch(
        Polygon(
            contour,
            closed=True,
            fill=False,
            edgecolor=shape_color_dict[shape_name],
            linewidth=2.5
        )
    )

# Create custom legend elements
legend_elements = [
    Line2D([0], [0], color=shape_color_dict[shape], lw=3, label=shape)
    for shape in shapes
]

# Formatting the plot
ax.axis('equal')
ax.set_xlabel('UMAP1', fontsize=12)
ax.set_ylabel('UMAP2', fontsize=12)
ax.set_title(f'Swedish Leaf Dataset Contours, SI={ss:.4f}', fontsize=24)

# Position the legend outside the plot
ax.legend(handles=legend_elements, title='Leaf Species', loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12)

fig.tight_layout()
plt.show()
```
![Leaf_UMAP_contour](../tutorials/Swedish_Leaf_data_results/Leafdata_UMAP_contour_clean.png)

The UMAP visualization demonstrates that the MO2GP effectively separates leaf species based on their morphological complexity and boundary frequencies. Distinctive shapes, such as Acer (maple-like) and Sorbus aucuparia, are isolated on the left, while the elongated Salix alba Sericea is partitioned to the bottom right. Conversely, leaf with similar shape profiles are clustered in close region, including the Sorbus intermedia/Quercus group and the Fagus silvatica/Salix cinerea pairing. Furthermore, a tight cluster formed by Salix aurita, Ulmus glabra, and Ulmus carpinifolia—along with four other species—highlight the MO2GP's ability to group leaves based on shared frequency details and overall contour signatures.

# MO2GP Parameter Optimization
In this tutorial, we adjusted MO2GP shape embedding parameter using the Swedish Leaf dataset. We evaluated values of **n_interp** ranging from 50 to 500, while using default parameter for other parameters, and the results are shown in the line graph. The silhouette index increased with larger **n_interp** values but reached a plateau at **n_interp** = 300. This indicates that increasing **n_interp** beyond this point does not lead to further improvement in clustering performance. Therefore, users should tune **n_interp** based on their specific dataset, as higher values do not necessarily correlate with a higher silhouette index. 
![n_interp_plot](../tutorials/Swedish_Leaf_data_results/n_interp_SI_plot.png)

Additionally, we examined the impact of the **n_smooth** parameter by testing values ranging from 0 to 8 and evaluating the results using the Silhouette Index (SI). While we observed a slight downward trend in the SI as **n_smooth** increased, this decrease was not really significant. This indicates that for this leaf dataset, additional smoothing does not substantially improve clustering performance. Therefore, we have set the default value for **n_smooth** to 0 in this tutorial.

![n_smooth_plot](../tutorials/Swedish_Leaf_data_results/SI_n_smooth_plot_leaf.png)

More detailed tutorials on additional datasets are available here:

[Simulation_Dataset](/README.md) | [MPEG7 Dataset](MPEG7_Dataset.md) |
[VeraFISH_Healthy_BMMC_Dataset](VeraFISH_Healthy_BMMC_dataset.md) 
