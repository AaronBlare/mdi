import numpy as np

def spearman_dist(data_1, data_2):
    res = np.sum((data_1 - data_2)**2)
    return res