# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 01:28:42 2016

@author: gjeziorski
"""

import numpy as np
import pandas as pd

from skimage import exposure, measure
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
matrices_thresholded = list(map(lambda matrix: matrix > 0.1, matrices_rescaled))
matrices_skeletonized = list(map(skeletonize, matrices_thresholded))
matrices_inverted = list(map(np.invert, matrices_skeletonized))
components = list(map(lambda image: measure.label(image, connectivity=1, return_num=True)[1], matrices_inverted))

xxx = measure.label(matrices_inverted[19], return_num=True)