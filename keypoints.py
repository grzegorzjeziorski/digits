# -*- coding: utf-8 -*-
from sklearn import svm
from sklearn.cluster import KMeans

from skimage.feature import corner_harris, corner_peaks, BRIEF

from utils import compute_histogram

def extract_descriptors(matrix, patch_size):
    extractor = BRIEF(descriptor_size=256, patch_size=patch_size)
    keypoints = corner_peaks(corner_harris(matrix), min_distance=1)
    extractor.extract(matrix, keypoints)
    return extractor.descriptors

def predict_clusters(kmeans, descriptors_single_image):
    if(len(descriptors_single_image) == 0):
        return []
    else:
        return kmeans.predict(descriptors_single_image)
    
def transform_to_histograms(input_data, patch_size, clusters_no):
    descriptors = list(map(lambda matrix: extract_descriptors(matrix, patch_size), input_data.matrices_rescaled))
    descriptors_float = list(map(lambda desc: desc.astype(float), descriptors))
    descriptors_flatened = [descriptor for descriptors_single in descriptors for descriptor in descriptors_single]
    kmeans = KMeans(init='random', n_clusters=clusters_no, n_init=10)
    kmeans.fit(descriptors_flatened)
    descriptors_mapped_to_clusters = list(map(lambda descriptors_single_image: predict_clusters(kmeans, descriptors_single_image), descriptors_float))
    descriptors_histograms = list(map(lambda elems: compute_histogram(elems, clusters_no), descriptors_mapped_to_clusters))
    return (kmeans, descriptors_histograms)
    
def transform_input_keypoints(matrix, kmeans):
    descriptors = extract_descriptors(matrix, 7)
    descriptors_float = descriptors.astype(float)
    return compute_histogram(kmeans.predict(descriptors_float), 100)
    
## best result with patch_size=7 and n=100
def keypoints_classifier(training_data):
    kmeans, histograms = transform_to_histograms(training_data, 7, 100)
    clf = svm.SVC()
    clf.fit(histograms, training_data.labels)
    return (kmeans, clf)

