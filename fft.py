# -*- coding: utf-8 -*-
import numpy as np

from scipy.fftpack import fft

from sklearn import svm

from data_loader import BasicImageTransformations

def transform_input_fft(matrix_as_array):
    bit = BasicImageTransformations(matrix_as_array)
    rows_summed_fft = fft(bit.rows_summed)
    columns_summed_fft = fft(bit.columns_summed)
    rows_summed_fft_real = np.real(rows_summed_fft)
    rows_summed_fft_imag = np.imag(rows_summed_fft)
    columns_summed_fft_real = np.real(columns_summed_fft)
    columns_summed_fft_imag = np.imag(columns_summed_fft)
    return np.concatenate((rows_summed_fft_real, rows_summed_fft_imag, columns_summed_fft_real, columns_summed_fft_imag))

def fft_classifier(training_data):
    ffts = list(map(transform_input_fft, training_data.matrices_as_arrays))
    clf = svm.SVC(kernel='poly', degree=3, C=1)
    clf.fit(ffts, training_data.labels)
    return clf