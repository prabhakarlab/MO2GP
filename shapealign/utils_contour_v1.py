import numpy as np
import cv2 as cv  # Fixed missing import
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import interpolate
from skimage.draw import polygon2mask
from sklearn.metrics import pairwise_distances

def carea(contour):
    if len(contour) < 3:
        return 0  # Avoid invalid areas for too few points
    return abs(cv.contourArea(contour.astype('float32')))

def ccenter(contour):
    return contour - np.mean(contour, axis=0)

def cclose(contour):
    return np.append(contour, [contour[0]], axis=0)

def corient(contour):
    contour = contour.copy()
    if not np.array_equal(contour[0], contour[-1]):
        contour = cclose(contour)  # Ensure closed
    x, y = contour[:, 0], contour[:, 1]
    area = np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]) + (x[-1] * y[0] - x[0] * y[-1])
    if area < 0:
        contour = np.flip(contour, axis=0)
    return contour

def cperimcum(contour):
    contour = cclose(contour)
    dx = np.diff(contour[:, 0])
    dy = np.diff(contour[:, 1])
    return np.insert(np.cumsum(np.sqrt(dx ** 2 + dy ** 2)), 0, 0)

def cperim(contour):
    return cperimcum(contour)[-1]

def edi(pt1, pt2, r=0.5):
    return r * (pt2 - pt1) + pt1

def csmooth(contour, n=0):
    if n == 0:
        return contour
    p = len(contour)
    for _ in range(n):
        contour_fwd = np.roll(contour, shift=1, axis=0)
        contour_bwd = np.roll(contour, shift=-1, axis=0)
        contour = contour / 2 + contour_fwd / 4 + contour_bwd / 4
    return contour

def remove_duplicate_points(contour):
    unique_points = list(dict.fromkeys(tuple(row) for row in contour))
    # Convert tuples back to lists
    return np.array([list(row) for row in unique_points])

def cinterpolate(contour, n=100, coarse=True, spline=True):
    """
    Interpolates a contour to have evenly spaced points.
    
    Parameters:
    -----------
    contour : np.ndarray
        Input contour as a 2D numpy array (N x 2).
    n : int
        Number of points after interpolation. Default is 100.
    coarse : bool
        Whether to perform initial coarse interpolation. Default is True.
    spline : bool
        Whether to apply B-spline smoothing. Default is True.
    
    Returns:
    -----------
    np.ndarray
        Interpolated contour.
    """
    if contour.shape[0] < 4:
        raise ValueError("Contour must have at least 4 points for interpolation.")

    # Ensure unique and valid points
    contour = remove_duplicate_points(contour)

    # Coarse interpolation (ensures uniform spacing)
    if coarse:
        orig = cperimcum(contour)
        targ = np.linspace(0, cperim(contour), n+1)
        cout = np.zeros((n, 2))
        cout[0] = contour[0]
        for i in range(1, n):
            valid_indices = np.where(orig <= targ[i])[0]
            k = valid_indices[-1] if len(valid_indices) > 0 else 0
            if orig[k] == orig[k+1]:  # Prevent division by zero
                cout[i] = contour[k]
            else:
                r = (targ[i] - orig[k]) / (orig[k+1] - orig[k])
                cout[i] = edi(contour[k % len(contour)], contour[(k+1) % len(contour)], r)
    else:
        cout = contour.copy()

    # Spline interpolation
    if spline:
        x, y = cout[:, 0], cout[:, 1]
        try:
            tck, u = interpolate.splprep([x, y], s=0, per=True)
            xi, yi = interpolate.splev(np.linspace(0, 1, n+1), tck)
            cout = np.column_stack((xi[:-1], yi[:-1]))
        except Exception as e:
            print(f"Warning: Spline interpolation failed: {e}. Skipping spline.")
            return cout  # Return without spline
    return cout


def preprocess(contour, n_interp=100, n_smooth=5, scale='area'):
    contour = cinterpolate(corient(ccenter(contour)), n=n_interp)
    if n_smooth > 0:
        contour = cinterpolate(csmooth(contour, n=n_smooth), n=n_interp)
    
    if scale == 'area':
        area = np.sqrt(carea(contour))
        if area > 0:
            contour /= area
    elif scale == 'perimeter':
        perim = cperim(contour)
        if perim > 0:
            contour /= perim
    contour = ccenter(contour)
    return contour
