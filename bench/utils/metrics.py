import os
import json
import matplotlib.pyplot as plt
import numpy as np
import warnings
from cycler import cycler

import matplotlib
from PIL import ImageColor

import pandas as pd

def get_projection_val(serie_x, serie_y, value):
    serie_sorted_x = serie_x[np.argsort(serie_x)]
    serie_sorted_y = serie_y[np.argsort(serie_x)]

    idx = np.searchsorted(serie_sorted_x, value)
    # assert idx!=0

    # if idx>=len(serie_x):
    # if idx==0 or idx>=len(serie_x):
    # if idx==0:
        # return None
    if idx==0:
        idx = 1
    if idx>=len(serie_x):
        idx = len(serie_x)-1

    
    diff = serie_sorted_x[idx] - serie_sorted_x[idx-1]
    diff2 = value - serie_sorted_x[idx-1]
    r = diff2/diff

    diff_y = serie_sorted_y[idx] - serie_sorted_y[idx-1]

    return serie_sorted_y[idx-1] + (r*diff_y)
    # return (serie_sorted_y[idx-1]+serie_sorted_y[idx])/2



def get_projection(serie_x, serie_y, values):
    return np.array([get_projection_val(serie_x, serie_y, v) for v in values])

def get_nne_rate(real_indices, indices, real_dist=None, dist=None,
                 random_state=0, max_k=32, verbose=0):

    if verbose >= 2:
        print("Precision \t|\t Recall")
    
    if verbose >= 1:
        iterator_1 = tqdm(zip(real_indices, indices))
    else:
        iterator_1 = zip(real_indices, indices)

    total_T = 0
    for x,y in iterator_1:
        T = len(set(x).intersection(set(y)))
        # T = len(set(x.reshape(-1).tolist()).intersection(set(y.reshape(-1).tolist())))
        total_T+=T

    N = len(real_indices)

    qnx = float(total_T)/(N * max_k)
    return qnx

    # rnx = ((N-1)*qnx-max_k)/(N-1-max_k)
    # return rnx


def create_recall_eps(eps):
    def get_recall_eps(real_indices, indices, real_dist, dist,
                    random_state=0, max_k=32, verbose=0):

        if verbose >= 2:
            print("Precision \t|\t Recall")
        
        if verbose >= 1:
            iterator_1 = tqdm(zip(real_indices, indices, real_dist, dist))
        else:
            iterator_1 = zip(real_indices, indices, real_dist, dist)

        total_T = 0
        for x,y,dx,dy in iterator_1:
            limiar_dist = np.max(dx)*(1.0+eps)
            T = np.sum(dy <= limiar_dist)
            total_T+=T

        N = len(real_indices)

        qnx = float(total_T)/(N * max_k)
        return qnx

    return get_recall_eps
