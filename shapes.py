# -*- coding: utf-8 -*-
import numpy as np

from scipy import stats

from sklearn import ensemble

from skimage import measure, feature

from data_loader import BasicImageTransformations

def convert_to_pixels_list(matrix):
    pixels = []
    for index, value in np.ndenumerate(matrix):
        if value > 0:        
            pixels.append(index)
    return pixels
    
def extract_dimension(pixels, axis):
    return list(map(lambda pixel: pixel[axis], pixels))
    
def transform_pixels_to_pair_of_lists(pixels):
    xs = extract_dimension(pixels, 1)
    ys = extract_dimension(pixels, 0)
    return (xs, ys)
    
def transform_pixels_with_function(pixels, func):
    return list(map(func, pixels))
    
def average_of_non_zero_elements(array):
    return stats.tmean(list(filter(lambda el: el > 0, array)))
    
def transform_input_shapes(matrix_as_array):
    bit = BasicImageTransformations(matrix_as_array)
    matrices_inverted = np.invert(bit.matrix_skeletonized)
    components = measure.label(matrices_inverted, connectivity=1, return_num=True)[1]    
    image_pixels = convert_to_pixels_list(bit.matrix_skeletonized)
    image_pixels_as_pair_of_lists = transform_pixels_to_pair_of_lists(image_pixels)
    pixels_x = extract_dimension(image_pixels, 1)
    pixels_y = extract_dimension(image_pixels, 0)
    min_x = min(pixels_x)
    max_x = max(pixels_x)
    min_y = min(pixels_y)
    max_y = max(pixels_y)
    mean_x = stats.tmean(pixels_x)
    mean_y = stats.tmean(pixels_y)
    variance_x = stats.tvar(pixels_x)
    variance_y = stats.tvar(pixels_y)
    correlation = stats.pearsonr(image_pixels_as_pair_of_lists[0], image_pixels_as_pair_of_lists[1])[0]
    xxy = transform_pixels_with_function(image_pixels, lambda pixel: pixel[0] * pixel[1] * pixel[1])
    xyy = transform_pixels_with_function(image_pixels, lambda pixel: pixel[0] * pixel[0] * pixel[1])
    mean_xxy = stats.tmean(xxy)  
    mean_xyy = stats.tmean(xyy)
    xy_tr2 = transform_pixels_with_function(image_pixels, lambda pixel: pixel[0] * pixel[1] * np.sin(pixel[0] / 2.0) * np.sin(pixel[1] / 2.0))
    xy_tr4 = transform_pixels_with_function(image_pixels, lambda pixel: pixel[0] * pixel[1] * np.sin(pixel[0] / 4.0) * np.sin(pixel[1] / 4.0))
    mean_xy_tr2 = stats.tmean(xy_tr2)  
    mean_xy_tr4 = stats.tmean(xy_tr4)
    print(mean_xy_tr2)
    print(mean_xy_tr4)
    edges = feature.canny(bit.matrix_float)
    edges_for_y = edges.sum(axis=1)
    edges_for_x = edges.sum(axis=0)
    avg_egdes_for_y = average_of_non_zero_elements(edges_for_y)
    avg_egdes_for_x = average_of_non_zero_elements(edges_for_x)
    skew_x = stats.skew(pixels_x)
    skew_y = stats.skew(pixels_y)
    features = np.array([components, min_x, max_x, min_y, max_y, mean_x, mean_y, variance_x, variance_y, correlation, mean_xxy, mean_xyy, mean_xy_tr2, mean_xy_tr4, avg_egdes_for_x, avg_egdes_for_y, skew_x, skew_y])
    return features

def shapes_classifier(training_data):
    matrices_as_arrays = training_data.matrices_as_arrays
    features = list(map(transform_input_shapes, matrices_as_arrays))
    gb_clf = ensemble.GradientBoostingClassifier(max_depth = 4, n_estimators = 500)
    gb_clf.fit(features, training_data.labels)
    return gb_clf