# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 23:50:46 2016

@author: gjeziorski
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dwt import dwt, dwt_distances, compute_dwt_label

from scipy import ndimage

from sklearn import svm
from sklearn import preprocessing
from sklearn import cross_validation
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics.pairwise import pairwise_distances

from scipy.cluster.hierarchy import fclusterdata

from skimage import exposure
from skimage.feature import hog
from skimage import data, color, exposure
from skimage import transform as tf
from skimage import img_as_float
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

rows_summed = list(map(lambda matrix: matrix.sum(axis=1), matrices_skeletonized))
columns_summed = list(map(lambda matrix: matrix.sum(axis=0), matrices_skeletonized))

#sanity check of data
plt.imshow(matrices[0], cmap = 'Greys_r')
plt.imshow(matrices[1], cmap = 'Greys_r')
plt.imshow(matrices[2], cmap = 'Greys_r')
plt.imshow(matrices[3], cmap = 'Greys_r')
plt.imshow(matrices[4], cmap = 'Greys_r')
plt.imshow(matrices[5], cmap = 'Greys_r')
plt.imshow(matrices[6], cmap = 'Greys_r')
plt.imshow(matrices[7], cmap = 'Greys_r')
plt.imshow(matrices[8], cmap = 'Greys_r')
plt.imshow(matrices[9], cmap = 'Greys_r')
plt.imshow(matrices[10], cmap = 'Greys_r')
plt.imshow(matrices[11], cmap = 'Greys_r')
plt.imshow(matrices[12], cmap = 'Greys_r')
plt.imshow(matrices[13], cmap = 'Greys_r')
plt.imshow(matrices[14], cmap = 'Greys_r')

skeleton_0 = skeletonize(matrices[0])
skeleton_1 = skeletonize(matrices[1])
skeleton_2 = skeletonize(matrices[2])

#more sanity checks
plt.imshow(matrices_float[0], cmap = 'Greys_r')
plt.imshow(matrices_float[1], cmap = 'Greys_r')
plt.imshow(matrices_float[2], cmap = 'Greys_r')
plt.imshow(matrices_float[3], cmap = 'Greys_r')
plt.imshow(matrices_float[4], cmap = 'Greys_r')

rescale_intensity(matrices_float[0])

#majority of digits is not turned (btw. then you might have a problem between 6 and 9)

scaled_arrays = preprocessing.scale(arrays)

rows_summed = map(lambda image: image.sum(axis = 1), matrices)
columns_summed = map(lambda image: image.sum(axis = 0), matrices)

rows_summed_and_colums_summed = list(map(lambda el: np.concatenate([el[0], el[1]]), zip(rows_summed, columns_summed)))
scaled_rows_summed_and_colums_summed = preprocessing.scale(rows_summed_and_colums_summed)

pca150 = PCA(n_components=150)
arrays150 = pca150.fit_transform(scaled_arrays)

clf = svm.SVC()
scores = cross_validation.cross_val_score(clf, arrays150, labels, cv=3, scoring='f1_weighted')

clf = svm.SVC()
scores = cross_validation.cross_val_score(clf, scaled_rows_summed_and_colums_summed, labels, cv=3, scoring='f1_weighted')

plt.imshow(matrices[1], cmap = 'Greys_r')
fd, hog_image = hog(matrices[1], orientations=4, pixels_per_cell=(4, 4), cells_per_block=(1, 1), visualise=True)
plt.imshow(hog_image, cmap= 'Greys_r')

def convert_to_hog(matrix):
    return hog(matrix, orientations=32, pixels_per_cell=(4, 4), cells_per_block=(1, 1), visualise=False)

def split_into_histograms(array):
    return np.split(array, 49)
    
def is_non_zero_hog(array):
    return len(list(filter(lambda x: x > 0.001, array))) > 0
    
def compute_histogram(elems, size):
    result = [0] * size
    for el in elems:
        result[el] = result[el] + 1
    return result

hogs = list(map(convert_to_hog, matrices_rescaled))
hogs_flatened = [histogram for histograms in map(split_into_histograms, hogs) for histogram in histograms]
non_zero_hogs = list(filter(is_non_zero_hog, hogs_flatened))

kmeans = KMeans(init='random', n_clusters=50, n_init=10)
kmeans.fit(non_zero_hogs)

hogs_split = list(map(split_into_histograms, hogs))
non_zero_hogs_split = [list(filter(is_non_zero_hog, hogs_for_picture)) for hogs_for_picture in hogs_split]
hogs_mapped_to_clusters = list(map(kmeans.predict, non_zero_hogs_split))
hogs_histograms = list(map(lambda elems: compute_histogram(elems, 50), hogs_mapped_to_clusters))

#sampling
hogs1000 = hogs[0:1000]
hogs1000_split = list(map(split_into_histograms, hogs1000))
hogs_flatened = []
for hogs_image in hogs1000_split:
    for hog in hogs_image:
        hogs_flatened.append(hog)
non_zero_hogs = list(filter(is_non_zero_hog, hogs_flatened))

kmeans = KMeans(init='random', n_clusters=10, n_init=10)
kmeans.fit(non_zero_hogs)
#sampling


clf = svm.SVC()
scores = cross_validation.cross_val_score(clf, hogs_histograms, labels, cv=3, scoring='f1_weighted')

#centering
matrices_center_of_mass = list(map(ndimage.measurements.center_of_mass, matrices_rescaled))
x_general = list(map(lambda tuple: tuple[1], matrices_center_of_mass))
y_general = list(map(lambda tuple: tuple[0], matrices_center_of_mass))
x_avg = sum(x_general) / len(x_general)
y_avg = sum(y_general) / len(y_general)

def align(matrix, x_avg, y_avg):
    (cy, cx) = ndimage.measurements.center_of_mass(matrix)
    translation = (cx - x_avg, cy - y_avg)
    transformation = tf.SimilarityTransform(translation=translation)
    return tf.warp(matrix, transformation)
    
aligned = list(map(lambda matrix: align(matrix, x_avg, y_avg), matrices_rescaled))
    
#translation = tf.SimilarityTransform(translation=(0.0, 5.1))
#matrices_0_translated = tf.warp(matrices_rescaled[0], translation)
#plt.imshow(matrices_rescaled[0], cmap = 'Greys_r')
#plt.imshow(matrices_0_translated, cmap = 'Greys_r')

tf.SimilarityTransform()

rows_summed_1000 = rows_summed[0:1000]
columns_summed_1000 = columns_summed[0:1000]

distances_rows = pairwise_distances(rows_summed_1000, metric=dwt)
distances_columns = pairwise_distances(columns_summed_1000, metric=dwt)

hierarchicalClusteringRows = AgglomerativeClustering(n_clusters=20, affinity=dwt, linkage='average')
hierarchicalClusteringRows.fit(rows_summed_1000)

hierarchicalClusteringColumns = AgglomerativeClustering(n_clusters=20, affinity=dwt, linkage='average')
hierarchicalClusteringColumns.fit(columns_summed_1000)





    