# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from skimage import exposure
from skimage.morphology import skeletonize

def find_minimum(dwt, i, j):
    list = []
    if i > 0:
        list.append(dwt[i - 1, j])
    if j > 0:
        list.append(dwt[i, j - 1])
    if i > 0 and j > 0:
        list.append(dwt[i - 1, j - 1])
    if len(list) == 0:
        return 0
    else:
        return min(list)
        
def dwt(first, second):
    n = len(first)
    m = len(second)
    dwt = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            cost = abs(first[i] - second[j]) #here you can enter limit on wrap size
            dwt[i, j] = cost + find_minimum(dwt, i, j)
    return dwt[n - 1, m - 1]

def dwt_distances(image, reference_rows_summed, reference_columns_summed, reference_labels):
    result = {}    
    rows_cost = {}
    columns_cost = {}
    for i in range(10):
        rows_cost[i] = 0
        columns_cost[i] = 0
    image_rows_summed = image.sum(axis = 1)
    image_columns_summed = image.sum(axis = 0)
    for i in range(len(reference_labels)):
        rows_difference = dwt(image_rows_summed, reference_rows_summed[i])
        columns_difference = dwt(image_columns_summed, reference_columns_summed[i])
        rows_cost[reference_labels[i]] = rows_cost[reference_labels[i]] + rows_difference
        columns_cost[reference_labels[i]] = columns_cost[reference_labels[i]] + columns_difference
    for i in range(10):
        rows_cost[i] = float(rows_cost[i]) / float(reference_labels.count(i))
        columns_cost[i] = float(columns_cost[i]) / float(reference_labels.count(i))
        result[i] = rows_cost[i] + columns_cost[i]
    return result
    
def compute_dwt_label(dwt_distances):
    return sorted(dwt_distances.items(), key = lambda x: x[1])[0][0]

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

correct = 0
errors = []
for i in range(1000, 1200):
    print(i)
    distances = dwt_distances(matrices_skeletonized[i], rows_summed[0:200], columns_summed[0:200], labels[0:200])
    if labels[i] == compute_dwt_label(distances):
        correct = correct + 1
    else:
        errors.append(compute_dwt_label(distances))        
        
print(float(correct) / float(200))
for i in range(10):
    print(str(i) + ": " + str(errors.count(i)))
    




