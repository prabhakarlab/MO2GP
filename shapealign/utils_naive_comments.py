"""
utils_naive.py

This module provides morphological feature extraction functions for 2D shape contours
represented as ordered point arrays (e.g., from OpenCV contours).

Each function computes a specific descriptor that quantifies geometric or
shape properties, typically for use in downstream shape analysis or machine learning.

All inputs are assumed to be NumPy arrays of shape (n_points, 2).
"""

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
    perimeter = cv.arcLength(mask, closed=True)
    area = cv.contourArea(mask)
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
    area = cv.contourArea(mask)
    hull = cv.convexHull(mask)
    convex_area = cv.contourArea(hull)
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
        float: the roundness of the shape
    """
    area = cv.contourArea(mask)
    (_,_), (width, height), _ = cv.minAreaRect(mask)
    return area / (width * height)

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
    (_, _), (minor_axis, major_axis), _ = cv.fitEllipse(mask)
    return np.sqrt(1 - (minor_axis ** 2 / major_axis ** 2))
       
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
    (_, _), (_, major_axis), _ = cv.fitEllipse(mask)
    return major_axis 

def minor_axis_length(mask):
    """
    Computes minor axis length of a fitted ellipse to the shape defined by points in `mask`.
    Args:
        mask (numpy.ndarray): array of size n x 2, where n is the number of points
            in the contour. The points are expected to be in order along the
            contour of the shape.
    Returns:
        float: the major axis length of the shape
    """
    (_, _), (minor_axis, _), _ = cv.fitEllipse(mask)
    return minor_axis 

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
    return np.sqrt(cv.contourArea(mask) / np.pi)

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
    (_, _), (width, height), _ = cv.minAreaRect(mask)
    return max(width, height) / min(width, height)


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
    area = cv.contourArea(mask)
    (_, _), (_, major_axis), _ = cv.fitEllipse(mask)
    return (4 * area) / (np.pi * major_axis ** 2)

