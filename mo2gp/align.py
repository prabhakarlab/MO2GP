from .utils_contour import *
from .utils_naive import *
from joblib import Parallel, delayed
from tqdm import tqdm
import numpy as np
from scipy.fft import fft
from copy import deepcopy as dcp
from sklearn.decomposition import PCA

class ShapeAlign:
    """
    A class for aligning and analyzing shape contours.
    
    Attributes:
    -----------
    raw_contours : list
        A list of raw input contours as numpy arrays.
    N : int 
        Total number of contours.
    contours : list
        Processed contours after interpolation and normalization.
    dft : np.ndarray
        Discrete Fourier Transform (DFT) representation of contours.
    scaled_mft : np.ndarray
        Magnitude-scaled feature representation of each contours.
    selected_features :np.ndarray
        Retrieves the specific feature from scaled_mftbased on the indices defined in idx.
    shape_embedding : np.ndarray
        Principal Component Analysis (PCA) embeddings of shape descriptors.
    descriptor : np.ndarray
        Computed shape descriptors.
    
    """

    def __init__(self, contours):
        """
        Initializes the ShapeAlign object with raw contours.

        Parameters:
        -----------
        contours : list
            A list of numpy arrays representing raw shape contours.
        """
        self.raw_contours = contours
        self.N = len(contours)
        self.contours = None        
        self.dft = None
        self.scaled_mft = None
        self.selected_features = None
        self.shape_embedding = None
        self.descriptor = None  # Initialize descriptor to avoid undefined variable error

    def __str__(self):
        """
        Returns a string representation of the ShapeAlign object.

        Returns:
        -----------
        str
            Summary of the number of contours and average points per contour.
        """
        s = f"ShapeAlign object\n"
        s += f"Number of contours: {self.N}\n"
        if self.contours is not None:
            s += f"Number of points per contour: {np.mean([c.shape[0] for c in self.contours])}\n"
        return s

    # Functions of contours --------------------------------------------------------------------------------
    
    @staticmethod
    def contour_simple_descriptors(contours, desc=['aspect_ratio', 'circularity', 'eccentricity',
                                                   'extent', 'roundness', 'solidity', 'area',]):
        """
        Computes shape descriptors for each contour in the input list.

        Parameters:
        -----------
        contours : list
            List of contours represented as numpy arrays of (x, y) coordinates.
        desc : list
            List of descriptor names to compute. Options include:
            - 'area', 'aspect_ratio', 'circularity', 'eccentricity',
              'extent', 'perimeter', 'roundness', 'solidity'

        Returns:
        -----------
        numpy.ndarray
            2D array where each row represents the shape descriptors for a contour.
        """
        descriptor_functions = {
            "area": area,
            "aspect_ratio": aspect_ratio,
            "circularity": circularity,
            "eccentricity": eccentricity,
            "extent": extent,
            "perimeter": perimeter,
            "roundness": roundness,
            "solidity": solidity
        }
        descriptors = np.array([
            [descriptor_functions[d](contours[i].astype('float32')) for d in desc]
            for i in tqdm(range(len(contours)), desc='Compute simple descriptors', position=0, leave=True)
        ]) 
        
        return descriptors

    # Main methods --------------------------------------------------------------------------------
    
    def preprocess_contours(
        self, 
        n_interp=250, 
        n_smooth=0, 
        scale='perimeter',
        show_progress=True, 
        num_workers=1
    ):
        """
        Preprocesses raw contours by interpolating.
    
        Parameters:
        -----------
        n_interp : int
            Number of points to interpolate each contour to. [Default = 250].
        n_smooth : int
            Number of times to apply smoothing. 0 means no smoothing. [Default = 0]. 
        scale : str
            Feature for scale-normalization ('perimeter' or 'area'). [Default = 'perimeter']
        show_progress : bool
            Whether to show a progress bar. [Default = True]
        num_workers : int
            Number of parallel workers. Defaults to 1.
        """
        disable_progress = not show_progress
        
        # Enable tqdm tracking for joblib
        if num_workers > 1:
            contours = Parallel(n_jobs=num_workers, backend='loky')(
                delayed(preprocess)(self.raw_contours[i], n_interp, n_smooth, scale) 
                for i in tqdm(range(self.N), desc='Preprocessing contours', disable=disable_progress, position=0, leave=True)
            )
        else:
            
            contours = [preprocess(self.raw_contours[i], n_interp, n_smooth, scale)
                        for i in tqdm(range(self.N), desc='Preprocessing contours', disable=disable_progress, position=0, leave=True)]
        
        self.contours = np.array(contours).astype(np.float64)
    
    def get_embedding(
        self, 
        get_descriptor=True,
        kernel=1,
        feature_select='variance',
        thrs=None,
        pcs=None,
        **args
    ):
        """
        Computes shape embeddings using Fourier Transform and PCA.

        Parameters:
        -----------
        get_descriptor : bool
            Whether to compute shape descriptors. Default is True.
        kernel : int
            Default is 1.
        feature_select : {'variance', 'mean'}
            Select the most relevant features of scaled_mft. 
            'variance' : Selects coefficients with variance above a threshold.
            'mean'     : Selects coefficients with a mean value above a threshold.
            Defaults to 'variance'.          
        thrs : float 
            Threshold for the 'feature_select'. 
            Default is None (0.01 for 'variance' and 0.05 or 'mean).
        pcs : int
            Number of principal components to retain. Defaults to 10.
        **args : dict
            Additional keyword arguments (e.g., specific descriptors).
        """
        
        if self.contours is None:
            raise ValueError("Contours must be preprocessed before embedding computation.")

        # Create complex contours
        contours = dcp(self.contours)
        x = contours[:, :, 0]  # Extract x coordinates
        y = contours[:, :, 1]  # Extract y coordinates
        complex_contours = (x + 1j * y)

        # Compute Fourier Transform and scale it
        dft = fft(complex_contours, axis=1)
        n_points = np.arange(dft.shape[1])
        A = (2 * np.pi * n_points) / len(n_points)
        if kernel == 1:
            # kernel = [1, 1, 0, -1, -1]
            W = 2 * 1j * (np.sin(A) + np.sin(2 * A))
        elif kernel == 2:
            # kernel = [1, 0, 0, 0, -1]
            W = 2 * 1j * np.sin(2 * A)
        elif kernel == 3:
            # kernel = [1, 0, -1]
            W = 2 * 1j * np.sin(A)
        elif kernel == 4:
            # kernel = [1, -1]
            W = np.cos(A) - 1 + 1j * np.sin(A)
        else:
            raise ValueError("Error: Only values [1, 2, 3, 4] are accepted for kernel.")
        features = np.sqrt(1 + np.abs(W)**2 + np.abs(W)**4)
        scaled_mft = features * np.abs(dft)

        # Feature select
        if feature_select == 'variance':
            temp = np.var(scaled_mft, axis=0)
            if thrs == None:
                thrs = 0.01
            idx = np.where(temp >= thrs)[0]
        elif feature_select == 'mean':
            temp = scaled_mft/np.max(scaled_mft)
            if thrs == None:
                thrs = 0.005
            idx = np.where(np.mean(temp, axis=0) >= thrs)[0]
        else:
            raise ValueError("Error: Only values [variance, mean] are accepted for feature_select.")
        selected_mft = scaled_mft[:,idx]

        ## Dimension reduction (PCA)
        pca = PCA(n_components=min(100, np.min(selected_mft.shape)))
        pca_data = pca.fit_transform(selected_mft)

        if pcs == None:
            var_rat = pca.explained_variance_ratio_
            var_diff = (var_rat[:(len(var_rat)-1)] - var_rat[1:(len(var_rat))])
            pcs = np.where(var_diff >= 0.001)[0]
            if len(pcs) > 0:
                pcs = pcs[-1] + 2
            else:
                pcs = len(var_rat)
        elif pcs > min(selected_mft.shape):
            print('Number of pcs is smaller than input dimension')
            print('Changing pcs to ', min(selected_mft.shape))
            pcs = min(selected_mft.shape)
        
        ## Compute Cells descriptors
        descriptors = None
        if get_descriptor:
            descriptors = self.contour_simple_descriptors(contours, **args)

        self.dft = dft
        self.scaled_mft = scaled_mft
        self.selected_features = idx
        self.shape_embedding = pca_data[:,:pcs]
        self.descriptor = descriptors
