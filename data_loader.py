# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from skimage import exposure
from skimage.morphology import skeletonize

class BasicImageTransformations:
    def __init__(self, matrix_as_array):
        self.matrix_as_array = matrix_as_array
        self.matrix = matrix_as_array.reshape(28, 28)
        self.matrix_float = self.matrix.astype(float)
        self.matrix_rescaled = exposure.rescale_intensity(self.matrix_float)
        self.matrix_thresholded = self.matrix_rescaled > 0.5
        self.matrix_skeletonized = skeletonize(self.matrix_thresholded)
        self.rows_summed = self.matrix.sum(axis=1)
        self.columns_summed = self.matrix.sum(axis=0)
        self.rows_summed_skeletonized = self.matrix_skeletonized.sum(axis=1)
        self.columns_summed_skeletonized = self.matrix_skeletonized.sum(axis=0)

class InputData:
    def __init__(self, matrices, matrices_float, matrices_rescaled, matrices_thresholded, matrices_skeletonized, matrices_as_arrays, labels, rows_summed, columns_summed):
        self.matrices = matrices
        self.matrices_float = matrices_float
        self.matrices_rescaled = matrices_rescaled
        self.matrices_thresholded = matrices_thresholded
        self.matrices_skeletonized = matrices_skeletonized
        self.matrices_as_arrays = matrices_as_arrays
        self.labels = labels
        self.rows_summed = rows_summed
        self.columns_summed = columns_summed

#extract everything except label and turns list of pixels into matrix
def extract_matrix(row, rows_no, columns_no):
    return np.array(row[1:]).reshape(rows_no, columns_no)
    
def extract_array(row):
    return np.array(row[1:])

def extract_label(row):
    return row[0]
    
def transform_data_frame(data_frame):
    matrices = list(map(lambda row: extract_matrix(row, 28, 28), data_frame.values))
    matrices_float = list(map(lambda matrix: matrix.astype(float), matrices))
    matrices_rescaled = list(map(exposure.rescale_intensity, matrices_float))
    matrices_thresholded = list(map(lambda matrix: matrix > 0.5, matrices_rescaled))
    matrices_skeletonized = list(map(skeletonize, matrices_thresholded))
    matrices_as_arrays = list(map(extract_array, data_frame.values))
    labels = list(map(extract_label, data_frame.values))
    rows_summed = list(map(lambda matrix: matrix.sum(axis=1), matrices_skeletonized))
    columns_summed = list(map(lambda matrix: matrix.sum(axis=0), matrices_skeletonized))
    return InputData(matrices, matrices_float, matrices_rescaled, matrices_thresholded, matrices_skeletonized, matrices_as_arrays, labels, rows_summed, columns_summed)
    
def load_data_frame(path):
    return pd.read_csv(path)
    
def trim_input_data(input_data, n):
    return InputData(input_data.matrices[0:n], input_data.matrices_float[0:n], input_data.matrices_rescaled[0:n], input_data.matrices_thresholded[0:n], input_data.matrices_skeletonized[0:n], input_data.matrices_as_arrays[0:n], input_data.labels[0:n], input_data.rows_summed[0:n], input_data.columns_summed[0:n])