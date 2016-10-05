# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 00:35:34 2016

@author: gjeziorski
"""

import numpy as np
import pandas as pd

from scipy.fftpack import fft

from sklearn import cross_validation
from sklearn import preprocessing
from sklearn import svm

from skimage import exposure
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

rows_summed = list(map(lambda image: image.sum(axis = 1), matrices_skeletonized))
columns_summed = list(map(lambda image: image.sum(axis = 0), matrices_skeletonized))

rows_summed_fft = list(map(fft, rows_summed))   
columns_summed_fft = list(map(fft, columns_summed))

rows_summed_fft_real = list(map(np.real, rows_summed_fft))
rows_summed_fft_imag = list(map(np.imag, rows_summed_fft))
columns_summed_fft_real = list(map(np.real, columns_summed_fft))
columns_summed_fft_imag = list(map(np.imag, columns_summed_fft))

ffts = list(map(lambda el: np.concatenate([el[0], el[1], el[2], el[3]]), zip(rows_summed_fft_real, rows_summed_fft_imag, columns_summed_fft_real, columns_summed_fft_imag)))

clf = svm.SVC()
scores = cross_validation.cross_val_score(clf, ffts, labels, cv=3, scoring='f1_weighted')
