# MO2GP Project
**Morphology O(2)-invariant General-purpose Projection (MO2GP)** is a shape embedding algorithm that is invariant to rotation, reflection, scale, and translation. MO2GP uses Fourier descriptors of complex-plane representations of contour position and regularity, with dimensionality reduction for noise reduction and scalability. It is highly efficient and suitable for large-scale morphological datasets. The pipeline is implemented in Python and can be run using Jupyter Notebooks or standard scripts.<br>

## 📥 How to Install

You can install **MO2GP** either directly from GitHub or by cloning the repository for development.

### Option 1 — Install Directly from GitHub

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
import numpy as np
from mo2gp import ShapeAlign  

# Example: Create a simple circle contour
circle = np.array([[np.cos(t), np.sin(t)] for t in np.linspace(0, 2*np.pi, 100, endpoint=False)])

shapes = ShapeAlign([circle])
shapes.preprocess_contours()
```

## Example Datasets
To demonstrate the functionalities of MO2GP, in this tutorial we utilize one simulation dataset, two widely recognized datasets, and one in-house spatial transcriptomics dataset:<br>
**1. Simulation dataset**<br>
**2. Swedish Leaf dataset**<br>
**3. MPEG-7 dataset**<br>
**4. VeraFISH Healthy BMMC dataset**<br>

# 1. Simulation dataset
First, we validated the MO2GP shape embedding using a synthetic dataset of 2,880 simulated shapes. The dataset includes 24 distinct categories derived from nine fundamental shapes: circle, ellipse, triangle, square, clover, pentagon, hexagon, kidney bean, nephroid, and three real biological shapes echinocyte cell, dendritic cell, and apoptotic cell; each generated at two aspect ratios (~1 and ~2). We also introduced low frequency noise to the contours and further expanded by subjecting each base shape to four orientation conditions: original, random rotation, vertical flip, and combined rotation with flipping.<br>


### Load the file 
```python
from mo2gp import datasets

simulation_data = datasets.load_simulation()

contour_input = simulation_data["contour"]
labels = simulation_data["labels"]
```
#### Contour of 24 simulated shapes
![Simulated_data_Contour](./tutorials/Simulation_data_results/Simulation_MO2GP_contour.png)
**Note:**  **boomerang** equal to **kidney bean** in the manuscript

### Run MO2GP analysis 
This step is where the MO2GP takes place. MO2GP Shape embedding uses the **ShapeAlign** class, which preprocess the raw shape outlines (contours) and performs advanced shape analysis using Fourier transforms and dimensionality reduction.<br>

First step is the **preprocess_contours**, a method to standardize all the contours to ensure all the contours are comparable. It processes the raw contours by centering, orienting (ensure the contour has clockwise orientation and is closed), interpolation, smoothing , and normalize the contour. <br>

The **preprocess_contours** step has several parameters, including **num_workers**, **n_interp**, **n_smooth**, and **scale**: <br>
1. *n_interp*   : Number of points to interpolate. Default is *250*. <br>
2. *n_smooth*   : Number of smoothing iterations to apply. Default is *0* (no smoothing). <br>
3. *scale*      : Method for scaling the contours to make them size-invariant. Options are *'perimeter'* or *'area'*. Default is *'perimeter'*.<br>
4. *num_workers*: The number of parallel workers to use for processing. Default is 1. <br>

Second step is the **get_embedding**, a method to processes the standardized shapes, extract the final embeddings and (optionally) calculate contours shape descriptors. It has several parameters including **get_descriptor**, **desc**, **kernel**, **feature_select**, **thrs**, and **pcs**. <br>
1. *get_descriptor*     : Whether to compute shape descriptors alongside embedding. Default is True. <br>
2. *desc*               : A list of specific descriptors to compute. Options are 'area', 'aspect_ratio', 'circularity', 'eccentricity', 'extent', 'perimeter', 'roundness', 'solidity'. If None, it will compute ['aspect_ratio', 'circularity', 'eccentricity', 'extent', 'roundness', 'solidity', 'area']. Default is None. <br>
3. *kernel*             : kernel used for spectral modulation to control smoothing and detail sensitivity. Options are [1='heavy smoothing',2='moderate smoothing',3='low smoothing',4='no smoothing']. Default is *1* (recommended to keep the default value).. <br>
4. *feature_select*     : Select the most relevant features {variance/mean}. Default is *variance*. <br>
5. *thrs*               : Numerical threshold utilized by *feature_select* method. If None, it defaults to *0.01* when using *variance*, and *0.05* when using *mean*. Defaults is *None*. <br>
6. *pcs*                : Number of principal components to retain. If None, the algorithm determines the optimal number based on variance. Defaults is *None*. <br>

Finally, the quality of the shape embeddings is evaluated using a modified silhouette score where we first find the average score for each specific group, then final silhouette score was then determined by taking the average of these per group means.

In this tutorial, we will use the default parameters.<br>

```python
import numpy as np
from sklearn.metrics import silhouette_samples
from mo2gp import ShapeAlign

# Start MO2GP shape analysis 
model_align = ShapeAlign(contours=contour_input)
model_align.preprocess_contours()
model_align.get_embedding()

shape_embedding = model_align.shape_embedding
contours = model_align.contours
descriptor = model_align.descriptor

def silhouette_score(dataIn, labels, metric='euclidean'):    
    output_sample = silhouette_samples(dataIn, labels, metric=metric)
    unique_labels = np.unique(labels)
    group_means = np.array([output_sample[labels == label].mean(axis=0) for label in unique_labels])
    return np.mean(group_means)

ss = silhouette_score(shape_embedding, labels, metric='euclidean')
print(f"Silhouette Score: {ss}, Embedding Shape: {shape_embedding.shape}")
```

### Visualize the UMAP
```python

import matplotlib.pyplot as plt
import umap

color_list = [
(0, 0, 0), # black
(0.788, 0.498, 0.498), # brown
(1.0, 0.647, 0.823), # hotpink
(0.701, 0.4, 0.701), # purple
(0.4, 0.4, 1.0), # blue
(0.4, 0.701, 0.4), # green
(0.456, 0.632, 0.779), #steel blue
(1.0, 0.788, 0.4), # orange
(1.0, 0.4, 0.4), # red
(0.0, 0.502, 0.502), # teal (for crenated circle)
(0.85, 0.15, 0.55), # deep magenta (for echinocyte)
(0.0, 0.8, 0.8), # bright cyan/teal (for dendritic cell)
(0.65, 0.85, 0.1) # chartreuse/lime yellow (for apoptotic cell)
]

def pastel(color, factor=0.5):
    # Mixes the color with white (1.0, 1.0, 1.0) based on the factor
    return tuple(1 - factor * (1 - c) for c in color)

color_pastel_list = [pastel(color, 0.4) for color in color_list]

shapes = [
       "circle",
       "triangle",
       "square",
       "clover",
       "pentagon",
       "star",
       "hexagon",
       "boomerang",
       "nephroid",
       "echinocyte",
       "dendritic cell",
       "apoptotic cell"
]

shapes_with_aspect = [shape + " aspect" for shape in shapes]

shape_color_dict = dict(zip(shapes, color_list))
shape_color_dict_2 = dict(zip(shapes_with_aspect, color_pastel_list))
shape_color_dict.update(shape_color_dict_2)
shape_color_dict

fit = umap.UMAP()
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
![Simulated_data_Contour](./tutorials/Simulation_data_results/Simulation_MO2GP_UMAP.png)

More detailed tutorials on additional datasets are available here:

[Swedish Leaf Dataset](./tutorials/Swedish_Leaf_Dataset.md) | [MPEG7 Dataset](./tutorials/MPEG7_Dataset.md) |
[VeraFISH_Healthy_BMMC_Dataset](./tutorials/VeraFISH_Healthy_BMMC_dataset.md) 


## ⚖️ Copyright & License
The codes here have been prepared by Jagadish Sankaran, Ignasius Joanito, Joseph Lee, and Shyam Prabhakar, Genome Institute of Singapore (GIS), Agency for Science, Technology and Research (A*STAR).

© 2022 - 2026 Agency for Science, Technology and Research (A*STAR).
