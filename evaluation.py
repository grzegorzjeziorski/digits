# -*- coding: utf-8 -*-
import numpy as np

from data_loader import load_data_frame, transform_data_frame, trim_input_data

from dwt import dwt_classifier
from shapes import transform_input_shapes, shapes_classifier
from pca_svm import transform_input_pca, pca_svm_classifier
from keypoints import transform_input_keypoints, keypoints_classifier
from fft import transform_input_fft, fft_classifier
from sklearn.cross_validation import train_test_split, cross_val_score 
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.grid_search import GridSearchCV
from sklearn import svm

data_frame = load_data_frame('~/Documents/digits/train.csv')

train_data_frame, test_data_frame = train_test_split(data_frame, test_size=0.6, random_state=123)

train_data = transform_data_frame(train_data_frame)
test_data = transform_data_frame(test_data_frame)
train_data_trimmed = trim_input_data(train_data, 1000)
test_data_trimmed = trim_input_data(test_data, 1000)

fft_clf = fft_classifier(train_data)
keypoints_kmeans, keypoints_clf = keypoints_classifier(train_data)
pca, svm_clf = pca_svm_classifier(train_data)
shapes_clf = shapes_classifier(train_data)

fft_input = list(map(transform_input_fft, test_data.matrices_as_arrays)) 
fft_score = fft_clf.score(fft_input, test_data.labels)
fft_results = fft_clf.predict(fft_input)

keypoints_input = list(map(lambda matrix: transform_input_keypoints(matrix, keypoints_kmeans), test_data.matrices_rescaled))
keypoints_score = keypoints_clf.score(keypoints_input, test_data.labels)
keypoints_results = keypoints_clf.predict(keypoints_input)

pca_svm_input = transform_input_pca(test_data.matrices_as_arrays, pca)
pca_svm_score = svm_clf.score(pca_svm_input, test_data.labels)
pca_svm_results = svm_clf.predict(pca_svm_input)

shapes_input = list(map(transform_input_shapes, test_data.matrices_as_arrays))
shapes_score = shapes_clf.score(shapes_input, test_data.labels)
shapes_results = shapes_clf.predict(shapes_input)

dwt_results = list(map(lambda matrix: dwt_classifier(matrix, train_data_trimmed), test_data_trimmed.matrices))

second_level_input = np.stack((fft_results, pca_svm_results, shapes_results), axis=-1)
encoder = OneHotEncoder()
second_level_input_encoded = encoder.fit_transform(second_level_input).toarray()

clf = svm.SVC(kernel='linear', C=1)
scores = cross_val_score(clf, second_level_input_encoded, test_data.labels, cv=10, scoring='f1_weighted')

parameters = {'kernel': ['rbf'], 'C':[0.1, 1, 1.5, 2, 2.5, 5, 10]}
svr = svm.SVC()
gs = GridSearchCV(svr, parameters)
gs.fit(second_level_input_encoded, test_data.labels)


gb_clf = GradientBoostingClassifier(max_depth = 4, n_estimators = 500)
gb_scores = cross_val_score(gb_clf, second_level_input_encoded, test_data.labels, cv=3, scoring='f1_weighted')

