import cv2 as cv
import numpy as np

'''
This module implements functions for computing morphological features from segmentation masks:
- Area
- Perimeter
- Circularity
- Solidity
- Extent
- Eccentricity
- Major axis length
- Minor axis length
- Max Feret diameter
- Min Feret diameter
- Equivalent diameter
- Aspect ratio
- Roundness
'''

def area(mask):
    """
    Computes area of a shape defined by points in `mask`.
    Args:
        mask (numpy.ndarray): array of size n x 2, where n is the number of points
            in the contour. The points are expected to be in order along the
            contour of the shape.
    Returns:
        float: the area of the shape
    """
    if mask is None or len(mask) < 3:
        return 0.0
    return cv.contourArea(mask)

def perimeter(mask):
    """
    Computes perimeter of a shape defined by points in `mask`.
    Args:
        mask (numpy.ndarray): array of size n x 2, where n is the number of points
            in the contour. The points are expected to be in order along the
            contour of the shape.
    Returns:
        float: the perimeter of the shape
    """
    if mask is None or len(mask) < 2:
        return 0.0
    return cv.arcLength(mask, closed=True)

def circularity(mask):
    """
    Computes circularity of a shape defined by points in `mask`.
    Circularity is defined as (4 * pi * area / perimeter ** 2). 
    Also known as form factor.
    Args:
        mask (numpy.ndarray): array of size n x 2, where n is the number of points
            in the contour. The points are expected to be in order along the
            contour of the shape.
    Returns:
        float: the circularity of the shape
    """
    if mask is None or len(mask) < 3:
        return np.nan
    perimeter = cv.arcLength(mask, closed=True)
    area = cv.contourArea(mask)
    if perimeter == 0:
        return np.nan
    return (4 * np.pi * area) / perimeter ** 2

def solidity(mask):
    """
    Computes solidity of a shape defined by points in `mask`.
    Solidity is defined as the ratio of the area of the shape to the area of its
    convex hull.
    Args:
        mask (numpy.ndarray): array of size n x 2, where n is the number of points
            in the contour. The points are expected to be in order along the
            contour of the shape.
    Returns:
        float: the solidity of the shape
    """
    if mask is None or len(mask) < 3:
        return np.nan
    area = cv.contourArea(mask)
    hull = cv.convexHull(mask)
    convex_area = cv.contourArea(hull)
    if convex_area == 0:
        return np.nan
    return area / convex_area

def extent(mask):
    """
    Computes extent of a shape defined by points in `mask`.
    Extent is defined as the ratio of the area of the shape to the area of its minimum rotated bounding box.
    Args:
        mask (numpy.ndarray): array of size n x 2, where n is the number of points
            in the contour. The points are expected to be in order along the
            contour of the shape.
            
    Returns:
        float: the extent of the shape
    """
    if mask is None or len(mask) < 3:
        return np.nan
    area = cv.contourArea(mask)
    (_,_), (width, height), _ = cv.minAreaRect(mask)
    bounding_area = width * height
    if bounding_area == 0:
        return np.nan
    return area / bounding_area

def eccentricity(mask):
    """
    Computes eccentricity of a shape defined by points in `mask`.
    Eccentricity is the ratio of the distance between the foci of the ellipse and its major axis length.
    Args:
        mask (numpy.ndarray): array of size n x 2, where n is the number of points
            in the contour. The points are expected to be in order along the
            contour of the shape.
            
    Returns:
        float: the eccentricity of the shape
    """
    if mask is None or len(mask) < 5:
        return np.nan
    try:
        (_, _), (minor_axis, major_axis), _ = cv.fitEllipse(mask)
        if major_axis == 0:
            return np.nan
        return np.sqrt(1 - (minor_axis ** 2 / major_axis ** 2))
    except cv.error:
        return np.nan

def major_axis_length(mask):
    """
    Computes major axis length of a fitted ellipse to the shape defined by points in `mask`.
    Args:
        mask (numpy.ndarray): array of size n x 2, where n is the number of points
            in the contour. The points are expected to be in order along the
            contour of the shape.
    Returns:
        float: the major axis length of the shape
    """
    if mask is None or len(mask) < 5:
        return np.nan
    try:
        (_, _), (_, major_axis), _ = cv.fitEllipse(mask)
        return major_axis
    except cv.error:
        return np.nan

def minor_axis_length(mask):
    """
    Computes minor axis length of a fitted ellipse to the shape defined by points in `mask`.
    Args:
        mask (numpy.ndarray): array of size n x 2, where n is the number of points
            in the contour. The points are expected to be in order along the
            contour of the shape.
    Returns:
        float: the minor axis length of the shape
    """
    if mask is None or len(mask) < 5:
        return np.nan
    try:
        (_, _), (minor_axis, _), _ = cv.fitEllipse(mask)
        return minor_axis
    except cv.error:
        return np.nan

def max_feret_diameter(mask):
    """
    Computes max feret diameter of the shape defined by points in `mask`.
    Args:
        mask (numpy.ndarray): array of size n x 2, where n is the number of points
            in the contour. The points are expected to be in order along the
            contour of the shape.
    Returns:
        float: the max Feret diameter of the shape
    """
    if mask is None or len(mask) < 2:
        return np.nan
    (_, _), (width, height), _ = cv.minAreaRect(mask)
    return max(width, height)

def min_feret_diameter(mask):
    """
    Computes min feret diameter of the shape defined by points in `mask`.
    Args:
        mask (numpy.ndarray): array of size n x 2, where n is the number of points
            in the contour. The points are expected to be in order along the
            contour of the shape.
    Returns:
        float: the min Feret diameter of the shape
    """
    if mask is None or len(mask) < 2:
        return np.nan
    (_, _), (width, height), _ = cv.minAreaRect(mask)
    return min(width, height)

def equivalent_diameter(mask):
    """
    Computes the equivalent diameter of the shape defined by the points in `mask`.
    Args:
        mask (numpy.ndarray): array of size n x 2, where n is the number of points
            in the contour. The points are expected to be in order along the
            contour of the shape.
    Returns:
        float: the equivalent diameter of the shape    
    """
    if mask is None or len(mask) < 3:
        return np.nan
    area = cv.contourArea(mask)
    return np.sqrt(4 * area / np.pi)

def aspect_ratio(mask):
    """
    Computes aspect ratio of a shape defined by points in `mask`.
    Aspect ratio is defined as the ratio of major axis to minor axis of the
    ellipse fitted to the shape.
    Args:
        mask (numpy.ndarray): array of size n x 2, where n is the number of points
            in the contour. The points are expected to be in order along the
            contour of the shape.
    Returns:
        float: the aspect ratio of the shape
    """
    if mask is None or len(mask) < 2:
        return np.nan
    (_, _), (width, height), _ = cv.minAreaRect(mask)
    min_dim = min(width, height)
    if min_dim == 0:
        return np.nan
    return max(width, height) / min_dim

def roundness(mask):
    """
    Computes roundness of a shape defined by points in `mask`.
    Roundness is defined as (4 * area) / (pi * major axis ** 2), where the area
    and major axis are of the ellipse fitted to the shape.
    Args:
        mask (numpy.ndarray): array of size n x 2, where n is the number of points
            in the contour. The points are expected to be in order along the
            contour of the shape.
            
    Returns:
        float: the roundness of the shape
    """
    if mask is None or len(mask) < 5:
        return np.nan
    try:
        area = cv.contourArea(mask)
        (_, _), (_, major_axis), _ = cv.fitEllipse(mask)
        if major_axis == 0:
            return np.nan
        return (4 * area) / (np.pi * major_axis ** 2)
    except cv.error:
        return np.nan
