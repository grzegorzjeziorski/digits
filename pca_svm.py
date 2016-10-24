# -*- coding: utf-8 -*-
from sklearn import svm
from sklearn.decomposition import PCA

def transform_input_pca(matrix_as_array, pca):
    return pca.transform(matrix_as_array)
    
def pca_svm_classifier(training_data):
    pca = PCA(n_components=150)
    pca.fit(training_data.matrices_as_arrays)
    arrays = pca.transform(training_data.matrices_as_arrays)
    clf = svm.SVC(kernel='poly', degree=3, C=1)
    clf.fit(arrays, training_data.labels)
    return (pca, clf)

