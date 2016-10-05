# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 00:01:21 2016

@author: gjeziorski
"""
import numpy as np
import pandas as pd

from scipy import ndimage

from sklearn.decomposition import PCA
from sklearn import cross_validation
from sklearn import preprocessing
from sklearn import svm

from skimage import exposure
from skimage import transform
from skimage.morphology import skeletonize

data_frame = pd.read_csv('~/Documents/digits/train.csv')

#extract everything except label and turns list of pixels into matrix
def extract_matrix(row, rows_no, columns_no):
    return np.array(row[1:]).reshape(rows_no, columns_no)
    
def extract_array(row):
    return np.array(row[1:])

def extract_label(row):
    return row[0]
    
matrices = list(map(lambda row: extract_matrix(row, 28, 28), data_frame.values))
matrices_float = list(map(lambda matrix: matrix.astype(float), matrices))
matrices_rescaled = list(map(exposure.rescale_intensity, matrices_float))
matrices_thresholded = list(map(lambda matrix: matrix > 0.5, matrices_rescaled))
matrices_skeletonized = list(map(skeletonize, matrices_thresholded))
arrays = list(map(extract_array, data_frame.values))
labels = list(map(extract_label, data_frame.values))
# this rescaling is not what we need. Alignment work is still valid
# this rescaling makes all images to sum to 0 and reducing variance.
# probably it is fine, but it is different than centeringw2
scaled_arrays = preprocessing.scale(arrays)

pca50 = PCA(n_components=50)
arrays50 = pca50.fit_transform(scaled_arrays)

clf = svm.SVC()
scores = cross_validation.cross_val_score(clf, arrays50, labels, cv=3, scoring='f1_weighted')

pca100 = PCA(n_components=100)
arrays100 = pca100.fit_transform(scaled_arrays)

clf = svm.SVC()
scores = cross_validation.cross_val_score(clf, arrays100, labels, cv=3, scoring='f1_weighted')

pca150 = PCA(n_components=150)
arrays150 = pca150.fit_transform(scaled_arrays[0:5000])

clf = svm.SVC()
scores = cross_validation.cross_val_score(clf, arrays150, labels[0:5000], cv=3, scoring='f1_weighted')

pca200 = PCA(n_components=200)
arrays200 = pca200.fit_transform(scaled_arrays)

clf = svm.SVC()
scores = cross_validation.cross_val_score(clf, arrays200, labels, cv=3, scoring='f1_weighted')

rows_summed = map(lambda image: image.sum(axis = 1), matrices_skeletonized)
columns_summed = map(lambda image: image.sum(axis = 0), matrices_skeletonized)

rows_summed_and_colums_summed = list(map(lambda el: np.concatenate([el[0], el[1]]), zip(rows_summed, columns_summed)))
scaled_rows_summed_and_colums_summed = preprocessing.scale(rows_summed_and_colums_summed)

clf = svm.SVC()
scores = cross_validation.cross_val_score(clf, scaled_rows_summed_and_colums_summed, labels, cv=3, scoring='f1_weighted')

#scaling to have the same top-bottom strech
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
    
matrices_float_rescaled = scale_matrices(matrices_float)

# alignment
matrices_center_of_mass = list(map(ndimage.measurements.center_of_mass, matrices_float_rescaled))
x_general = list(map(lambda tuple: tuple[1], matrices_center_of_mass))
y_general = list(map(lambda tuple: tuple[0], matrices_center_of_mass))
x_avg = sum(x_general) / len(x_general)
y_avg = sum(y_general) / len(y_general)

def align(matrix, x_avg, y_avg):
    (cy, cx) = ndimage.measurements.center_of_mass(matrix)
    translation = (cx - x_avg, cy - y_avg)
    transformation = transform.SimilarityTransform(translation=translation)
    return transform.warp(matrix, transformation)
    
rescaled_aligned = list(map(lambda matrix: align(matrix, x_avg, y_avg), matrices_float_rescaled))
rescaled_aligned_arrays = list(map(lambda matrix: np.reshape(matrix, (1, 28 * 28))[0], rescaled_aligned))
rescaled_aligned_arrays = preprocessing.scale(rescaled_aligned_arrays)

pca50 = PCA(n_components=50)
arrays50 = pca50.fit_transform(rescaled_aligned_arrays)

pca100 = PCA(n_components=100)
arrays100 = pca100.fit_transform(rescaled_aligned_arrays)

pca150 = PCA(n_components=150)
arrays150 = pca150.fit_transform(rescaled_aligned_arrays)

clf = svm.SVC()
scores = cross_validation.cross_val_score(clf, arrays50, labels, cv=3, scoring='f1_weighted')

clf = svm.SVC()
scores = cross_validation.cross_val_score(clf, arrays100, labels, cv=3, scoring='f1_weighted')

clf = svm.SVC()
scores = cross_validation.cross_val_score(clf, arrays150, labels, cv=3, scoring='f1_weighted')
