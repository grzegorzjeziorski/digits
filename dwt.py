# -*- coding: utf-8 -*-
import numpy as np

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

def dwt_classifier(matrix, training_data):
    distances = dwt_distances(matrix, training_data.rows_summed, training_data.columns_summed, training_data.labels)
    return sorted(distances.items(), key = lambda x: x[1])[0][0]    
    
    

    




