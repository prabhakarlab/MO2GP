"""
utils_contour.py

This module provides contour preprocessing utilities for shape analysis,
including centering, orientation, smoothing, interpolation, and normalization.

These operations are designed to standardize 2D shape contours so that
they can be compared and analyzed in downstream pipelines (e.g., PCA, FFT).
"""

import numpy as np
import cv2 as cv  # OpenCV for contour geometry
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import interpolate
from skimage.draw import polygon2mask
from sklearn.metrics import pairwise_distances

def carea(contour):
    """
    Computes the area enclosed by a 2D contour.

    Parameters:
        contour : np.ndarray
            Contour of shape (n_points, 2)

    Returns:
        float
            Absolute area of the contour.
    """
    if len(contour) < 3:
        return 0  # A polygon must have at least 3 points to have an area
    return abs(cv.contourArea(contour.astype('float32')))

def ccenter(contour):
    """
    Centers the contour at the origin by subtracting its mean.

    Parameters:
        contour : np.ndarray

    Returns:
        np.ndarray
            Centered contour (same shape).
    """
    return contour - np.mean(contour, axis=0)

def cclose(contour):
    """
    Closes the contour by appending the first point at the end if not closed.

    Parameters:
        contour : np.ndarray

    Returns:
        np.ndarray
            Closed contour (n+1, 2)
    """
    return np.append(contour, [contour[0]], axis=0)

def corient(contour):
    """
    Ensures the contour has clockwise orientation and is closed.

    Parameters:
        contour : np.ndarray

    Returns:
        np.ndarray
            Oriented and closed contour.
    """
    contour = contour.copy()
    if not np.array_equal(contour[0], contour[-1]):
        contour = cclose(contour)
    x, y = contour[:, 0], contour[:, 1]
    # Compute signed area using shoelace formula
    area = np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]) + (x[-1] * y[0] - x[0] * y[-1])
    if area < 0:
        contour = np.flip(contour, axis=0)  # Make it clockwise
    return contour

def cperimcum(contour):
    """
    Computes cumulative perimeter (arc length) at each point of the contour.

    Parameters:
        contour : np.ndarray

    Returns:
        np.ndarray
            Cumulative distance array of shape (n_points + 1,)
    """
    contour = cclose(contour)
    dx = np.diff(contour[:, 0])
    dy = np.diff(contour[:, 1])
    return np.insert(np.cumsum(np.sqrt(dx ** 2 + dy ** 2)), 0, 0)

def cperim(contour):
    """
    Computes total perimeter (arc length) of the closed contour.

    Parameters:
        contour : np.ndarray

    Returns:
        float
            Total perimeter.
    """
    return cperimcum(contour)[-1]

def edi(pt1, pt2, r=0.5):
    """
    Computes a point along the edge from pt1 to pt2 at ratio r.

    Parameters:
        pt1, pt2 : np.ndarray
        r : float

    Returns:
        np.ndarray
            Interpolated point.
    """
    return r * (pt2 - pt1) + pt1

def csmooth(contour, n=0):
    """
    Applies forward-backward smoothing to contour points.

    Parameters:
        contour : np.ndarray
        n : int
            Number of smoothing iterations.

    Returns:
        np.ndarray
            Smoothed contour.
    """
    if n == 0:
        return contour
    p = len(contour)
    for _ in range(n):
        contour_fwd = np.roll(contour, shift=1, axis=0)
        contour_bwd = np.roll(contour, shift=-1, axis=0)
        contour = contour / 2 + contour_fwd / 4 + contour_bwd / 4
    return contour

def remove_duplicate_points(contour):
    """
    Removes duplicate points from the contour.

    Parameters:
        contour : np.ndarray

    Returns:
        np.ndarray
            Deduplicated contour.
    """
    unique_points = list(dict.fromkeys(tuple(row) for row in contour))
    return np.array([list(row) for row in unique_points])

def cinterpolate(contour, n=100, coarse=True, spline=True):
    """
    Interpolates a contour to have evenly spaced points.

    Parameters:
        contour : np.ndarray
            Input contour of shape (n_points, 2)
        n : int
            Number of interpolated points.
        coarse : bool
            Whether to apply linear interpolation first.
        spline : bool
            Whether to use B-spline fitting after linear interpolation.

    Returns:
        np.ndarray
            Interpolated contour of shape (n, 2)
    """
    if contour.shape[0] < 4:
        raise ValueError("Contour must have at least 4 points for interpolation.")

    contour = remove_duplicate_points(contour)

    # Step 1: Coarse linear interpolation
    if coarse:
        orig = cperimcum(contour)
        targ = np.linspace(0, cperim(contour), n + 1)
        cout = np.zeros((n, 2))
        cout[0] = contour[0]
        for i in range(1, n):
            valid_indices = np.where(orig <= targ[i])[0]
            k = valid_indices[-1] if len(valid_indices) > 0 else 0
            if orig[k] == orig[k + 1]:  # avoid division by zero
                cout[i] = contour[k]
            else:
                r = (targ[i] - orig[k]) / (orig[k + 1] - orig[k])
                cout[i] = edi(contour[k % len(contour)], contour[(k + 1) % len(contour)], r)
    else:
        cout = contour.copy()

    # Step 2: B-spline smoothing (optional)
    if spline:
        x, y = cout[:, 0], cout[:, 1]
        try:
            tck, u = interpolate.splprep([x, y], s=0, per=True)
            xi, yi = interpolate.splev(np.linspace(0, 1, n + 1), tck)
            cout = np.column_stack((xi[:-1], yi[:-1]))
        except Exception as e:
            print(f"Warning: Spline interpolation failed: {e}. Skipping spline.")
            return cout

    return cout

def preprocess(contour, n_interp=100, n_smooth=5, scale='area'):
    """
    Applies full preprocessing pipeline:
    - Centering
    - Orientation
    - Interpolation
    - Smoothing
    - Normalization (by area or perimeter)

    Parameters:
        contour : np.ndarray
        n_interp : int
            Number of interpolated points.
        n_smooth : int
            Number of smoothing steps.
        scale : {'area', 'perimeter'}
            Normalization method.

    Returns:
        np.ndarray
            Preprocessed contour.
    """
    # Step 1: Normalize orientation and center
    contour = cinterpolate(corient(ccenter(contour)), n=n_interp)

    # Step 2: Optional smoothing
    if n_smooth > 0:
        contour = cinterpolate(csmooth(contour, n=n_smooth), n=n_interp)

    # Step 3: Normalize size
    if scale == 'area':
        area = np.sqrt(carea(contour))
        if area > 0:
            contour /= area
    elif scale == 'perimeter':
        perim = cperim(contour)
        if perim > 0:
            contour /= perim

    # Step 4: Center again after scaling
    contour = ccenter(contour)

    return contour
