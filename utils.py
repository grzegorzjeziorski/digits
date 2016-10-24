# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 00:57:58 2016

@author: gjeziorski
"""

from scipy import ndimage
from skimage import transform

def compute_histogram(elems, size):
    result = [0] * size
    for el in elems:
        result[el] = result[el] + 1
    return result
    
def normalize_histogram(elems):
    cumsum = float(sum(elems))
    elems_float = list(map(float, elems))
    return list(map(lambda elem: elem / cumsum, elems_float))
    
def find_min_max(matrix):
    minimum = len(matrix)
    maximum = 0
    for y_idx in range(len(matrix)):
        for x_idx in range(len(matrix[0])):
            if matrix[y_idx][x_idx] > 0:
                minimum = min(minimum, y_idx)
                maximum = max(maximum, y_idx)
    return (minimum, maximum)

def scale_matrices(matrices):
    result = []
    matrices_stretch = list(map(find_min_max, matrices))
    lengths = list(map(lambda el: el[1] - el[0], matrices_stretch))
    avg_length = float(sum(lengths)) / float(len(lengths))
    for idx in range(len(matrices)):
        rescaled = transform.rescale(matrices[idx], avg_length / float(lengths[idx]))
        result.append(rescaled[0:28, 0:28])
    return result
    
def align(matrix, x_avg, y_avg):
    (cy, cx) = ndimage.measurements.center_of_mass(matrix)
    translation = (cx - x_avg, cy - y_avg)
    transformation = transform.SimilarityTransform(translation=translation)
    return transform.warp(matrix, transformation)

