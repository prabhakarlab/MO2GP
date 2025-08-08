from utils_contour import *
from utils_naive import *
from joblib import Parallel, delayed
from tqdm import tqdm
import numpy as np
from scipy.fft import fft
from sklearn.decomposition import PCA
import warnings

class ShapeAlign:
    """
    A class for aligning, analyzing, and embedding shape contours.

    This class provides a complete pipeline for processing raw shape outlines (contours),
    from preprocessing (interpolation, smoothing) to advanced shape analysis using
    Fourier transforms and Principal Component Analysis (PCA).

    Attributes:
    -----------
    raw_contours : list
        A list of the original, unprocessed input contours. Each contour is a NumPy array
        of shape (n_points, 2).
    N : int
        The total number of contours.
    contours : np.ndarray | None
        A NumPy array of shape (N, n_interp, 2) containing the processed contours
        after interpolation, smoothing, and normalization. This is populated after
        running the `preprocess_contours` method.
    shape_embedding : np.ndarray | None
        A NumPy array of shape (N, pcs) containing the PCA embeddings of the shape
        descriptors. This represents the shapes in a lower-dimensional space and is
        populated by the `get_embedding` method.
    descriptor : np.ndarray | None
        A NumPy array of shape (N, n_descriptors) containing simple morphological
        shape descriptors (e.g., area, circularity). This is populated by the
        `get_embedding` method if `get_descriptor` is True.
    """

    def __init__(self, contours):
        """
        Initializes the ShapeAlign object with raw contours.

        Parameters:
        -----------
        contours : list
            A list of raw input contours, where each contour is a NumPy array of
            shape (n_points, 2) representing vertex coordinates.
        """
        self.raw_contours = contours
        self.N = len(contours)
        self.contours = None
        self.shape_embedding = None
        self.descriptor = None


    def __str__(self):
        """
        Returns a string representation of the ShapeAlign object.
        """
        s = f"ShapeAlign object\n"
        s += f"Number of contours: {self.N}\n"
        if self.contours is not None:
            s += f"Contours processed: Yes (Average points: {np.mean([c.shape[0] for c in self.contours]):.2f})\n"
        else:
            s += f"Contours processed: No\n"
        if self.shape_embedding is not None:
            s += f"Shape embedding generated: Yes (Dimensions: {self.shape_embedding.shape[1]})\n"
        else:
            s += f"Shape embedding generated: No\n"
        return s

    @staticmethod
    def contour_simple_descriptors(contours, desc=('aspect_ratio', 'circularity', 'eccentricity',
                                                   'extent', 'roundness', 'solidity', 'area')):
        """
        Computes simple morphological descriptors for each contour.

        This is a static method that can be used independently. The descriptor
        functions are assumed to be imported from `utils_naive`.

        Parameters:
        -----------
        contours : list or np.ndarray
            A list or array of contours to be analyzed.
        desc : tuple[str]
            A tuple of strings specifying which descriptors to compute. Available
            options depend on the functions imported from `utils_naive`.

        Returns:
        --------
        np.ndarray
            An array of shape (n_contours, n_descriptors) containing the computed
            descriptor values.
        """
        # Dictionary mapping descriptor names to their respective functions.
        descriptor_functions = {
            "area": area, "aspect_ratio": aspect_ratio, "circularity": circularity,
            "eccentricity": eccentricity, "extent": extent, "perimeter": perimeter,
            "roundness": roundness, "solidity": solidity
        }
        
        # Compute the specified descriptors for each contour.
        descriptors = np.array([
            [descriptor_functions[d](c.astype('float32')) for d in desc]
            for c in tqdm(contours, desc='Compute simple descriptors', position=0, leave=True)
        ])
        return descriptors

    def preprocess_contours(
        self,
        n_interp=250,
        n_smooth=0,
        scale='perimeter',
        show_progress=True,
        num_workers=1
    ):
        """
        Preprocesses raw contours by interpolating, smoothing, and scaling them.

        This method standardizes the contours to ensure they are comparable.
        Processing can be parallelized by setting `num_workers` > 1.

        Parameters:
        -----------
        n_interp : int, optional
            The number of points to interpolate for each contour, resulting in
            contours of a uniform size. Default is 250.
        n_smooth : int, optional
            The number of smoothing iterations to apply. If 0, no smoothing is
            performed. Default is 0.
        scale : str, optional
            The method for scaling the contours to make them size-invariant.
            Currently only accept 'perimeter' or 'area'. Default is 'perimeter'.
        show_progress : bool, optional
            If True, a `tqdm` progress bar is displayed. Default is True.
        num_workers : int, optional
            The number of parallel workers to use for processing. If 1, processing
            is done sequentially. Default is 1.
        """
        disable_progress = not show_progress

        # The core processing is handled by the `preprocess` function from `.utils_contour`.
        # We can parallelize this step for efficiency with large datasets.
        if num_workers > 1:
            contours = Parallel(n_jobs=num_workers, backend='loky')(
                delayed(preprocess)(self.raw_contours[i], n_interp, n_smooth, scale)
                for i in tqdm(range(self.N), desc='Preprocessing contours', disable=disable_progress, position=0, leave=True)
            )
        else:
            contours = [preprocess(c, n_interp, n_smooth, scale)
                        for c in tqdm(self.raw_contours, desc='Preprocessing contours', disable=disable_progress, position=0, leave=True)]
        
        # Store the processed contours as a NumPy array with a consistent data type.
        self.contours = np.array(contours).astype(np.float64)

    def get_embedding(
        self,
        pcs=None,
        get_descriptor=True,
        feature_select='variance',
        thrs=None,
        **args
    ):
        """
        Computes shape embeddings from preprocessed contours using a Fourier-based method.

        The process involves:
        1. Representing contours as complex numbers.
        2. Applying the Fast Fourier Transform (FFT).
        3. Scaling Fourier coefficients to enhance important features.
        4. Selecting the most informative coefficients (features).
        5. Applying Principal Component Analysis (PCA) for dimensionality reduction.

        Parameters:
        -----------
        pcs : int | None, optional
            The final number of principal components (dimensions) for the embedding.
            If None, the number is determined automatically based on the variance
            explained by the components. Default is None.
        get_descriptor : bool, optional
            If True, also computes and stores simple morphological descriptors.
            Default is True.
        feature_select : {'variance', 'mean'}, optional
            The method used to select Fourier coefficients before PCA.
            - 'variance': Selects coefficients with variance above a threshold.
            - 'mean': Selects coefficients with a mean value above a threshold.
            Default is 'variance'.
        thrs : float | None, optional
            The threshold for the `feature_select` method. If None, a default
            value is used (0.01 for 'variance', 0.005 for 'mean').
        **args : dict, optional
            Additional keyword arguments to be passed to `contour_simple_descriptors`,
            such as `desc` to specify which descriptors to compute.
        """
        if self.contours is None:
            raise ValueError("Contours must be preprocessed. Please run .preprocess_contours() first.")

        # --- Step 1: Convert to Complex Numbers ---
        # Represent each (x, y) coordinate as a complex number x + iy.
        data = self.contours[:, :, 0] + 1j * self.contours[:, :, 1]

        # --- Step 2: Compute Fourier Transform ---
        # The FFT decomposes the shape's boundary into a sum of sinusoids (epicycles).
        data = fft(data, axis=1)

        # --- Step 3: Calculate and Apply a Weighted Scaling Factor ---
        # This custom scaling emphasizes certain frequencies, making the features more robust.
        # It's a form of normalization that can improve PCA results.
        n_points = data.shape[1]
        A = (2 * np.pi * np.arange(n_points)) / n_points
        W = 2 * 1j * (np.sin(A) + np.sin(2 * A))
        scaling_features = np.sqrt(1 + np.abs(W)**2 + np.abs(W)**4)
        # We use the magnitude |data| to make the descriptor invariant to the starting point of the contour.
        data = scaling_features * np.abs(data)

        # --- Step 4: Feature Selection ---
        # Reduce dimensionality by keeping only the most informative Fourier coefficients.
        if feature_select == 'variance':
            if thrs is None: thrs = 0.01
            selected_indices = np.where(np.var(data, axis=0) > thrs)[0]
        elif feature_select == 'mean':
            if thrs is None: thrs = 0.005
            selected_indices = np.where(np.mean(data / np.max(data), axis=0) > thrs)[0]
        else:
            raise ValueError("Error: Only values ['variance', 'mean'] are accepted for feature_select.")
        
        # Check if any features were selected. If not, PCA will fail.
        if len(selected_indices) == 0:
            raise ValueError(f"No features were selected with feature_select='{feature_select}' and thrs={thrs}. Try lowering the threshold.")
        
        data = data[:, selected_indices]

        # --- Step 5: Principal Component Analysis (PCA) ---
        # Further reduce dimensionality and find the primary axes of variation in the dataset.
        num_components = min(100, np.min(data.shape)) # Cap components for stability.
        pca = PCA(n_components=num_components)
        data = pca.fit_transform(data)

        # --- Step 6: Determine Final Number of Components ---
        # If the user doesn't specify `pcs`, we automatically find the 'elbow' point.
        if pcs is None:
            explained_variance_ratio = pca.explained_variance_ratio_
            variance_diff = np.diff(explained_variance_ratio)
            # Find where the drop in explained variance becomes small.
            significant_pcs_indices = np.where(np.abs(variance_diff) > 0.001)[0]
            # Use a fallback if no significant drops are found.
            if len(significant_pcs_indices) > 0:
                # Add 2 for the last significant component and the one after it.
                pcs = significant_pcs_indices[-1] + 2
            else:
                # Fallback to a reasonable number or all components if few are available.
                pcs = min(10, num_components)
        elif pcs > data.shape[1]:
            warnings.warn(f"Requested pcs ({pcs}) > available components ({data.shape[1]}). Using {data.shape[1]}.")
            pcs = data.shape[1]

        # --- Step 7: Final Assignments ---
        # Store the final embedding and compute simple descriptors if requested.
        self.shape_embedding = data[:, :pcs]

        if get_descriptor:
            desc_list = args.get('desc', ('aspect_ratio', 'circularity', 'eccentricity',
                                          'extent', 'roundness', 'solidity', 'area'))
            self.descriptor = self.contour_simple_descriptors(self.contours, desc=desc_list)