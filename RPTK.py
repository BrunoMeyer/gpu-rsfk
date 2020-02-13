"""Bindings for the Barnes Hut TSNE algorithm with fast nearest neighbors

Refs:
References
[1] van der Maaten, L.J.P.; Hinton, G.E. Visualizing High-Dimensional Data
Using t-SNE. Journal of Machine Learning Research 9:2579-2605, 2008.
[2] van der Maaten, L.J.P. t-Distributed Stochastic Neighbor Embedding
http://homepage.tudelft.nl/19j49/t-SNE.html

"""

import numpy as np
import ctypes
import os
import pkg_resources

def ord_string(s):
    b = bytearray()
    arr = b.extend(map(ord, s))
    return np.array([x for x in b] + [0]).astype(np.uint8)

class RPTK(object):
    def __init__(self,
                 num_nearest_neighbors
            ):
        self.num_nearest_neighbors = num_nearest_neighbors
        
        # Build the hooks for the BH T-SNE library
        self._path = pkg_resources.resource_filename('RPTK','') # Load from current location
        # self._faiss_lib = np.ctypeslib.load_library('libfaiss', self._path) # Load the ctypes library
        # self._gpufaiss_lib = np.ctypeslib.load_library('libgpufaiss', self._path) # Load the ctypes library
        self._lib = np.ctypeslib.load_library('librptk', self._path) # Load the ctypes library
        
        # Hook the BH T-SNE function
        self._lib.pymodule_rptk_knn.restype = None
        self._lib.pymodule_rptk_knn.argtypes = [ 
                ctypes.c_int, # number of trees
                ctypes.c_int, # number of nearest neighbors
                ctypes.c_int, # total of points
                ctypes.c_int, # dimensions of points
                ctypes.c_int, # maximum depth of tree
                ctypes.c_int, # verbose (1,2,3 or 4)
                np.ctypeslib.ndpointer(np.float32, ndim=2, flags='ALIGNED, CONTIGUOUS'), # points
                np.ctypeslib.ndpointer(np.int32, ndim=2, flags='ALIGNED, CONTIGUOUS, WRITEABLE'), # knn-indices
                np.ctypeslib.ndpointer(np.float32, ndim=2, flags='ALIGNED, F_CONTIGUOUS, WRITEABLE'), # knn-sqd-distances
                # TODO: run name - Char pointer?
                ]

    def find_nearest_neighbors(self, points, n_trees=1, max_tree_depth=500, verbose=1):

        N = points.shape[0]
        D = points.shape[1]
        K = self.num_nearest_neighbors
        
        self.points = np.require(points, np.float32, ['CONTIGUOUS', 'ALIGNED'])
        self._knn_indices = np.require(np.full((N,K), -1), np.int32, ['CONTIGUOUS', 'ALIGNED', 'WRITEABLE'])
        self._knn_squared_dist = np.require(np.full((N,K), np.inf), np.float32, ['F_CONTIGUOUS', 'ALIGNED', 'WRITEABLE'])


        self._lib.pymodule_rptk_knn(
                ctypes.c_int(n_trees), # dimensions of projection
                ctypes.c_int(K), # dimensions of projection
                ctypes.c_int(N), # dimensions of projection
                ctypes.c_int(D), # dimensions of projection
                ctypes.c_int(max_tree_depth), # dimensions of projection
                ctypes.c_int(verbose), # dimensions of projection
                self.points,
                self._knn_indices,
                self._knn_squared_dist)

        return self._knn_indices, self._knn_squared_dist


def get_nne_rate(h_indices, l_indices, random_state=0, max_k=32,
                 verbose=0):
    


    if verbose >= 2:
        print("Precision \t|\t Recall")
    
    if verbose >= 1:
        iterator_1 = tqdm(zip(h_indices, l_indices))
    else:
        iterator_1 = zip(h_indices, l_indices)

    total_T = 0
    for x,y in iterator_1:
        T = len(set(x).intersection(set(y)))
        total_T+=T

    N = len(h_indices)
    qnx = float(total_T)/(N * max_k)

    rnx = ((N-1)*qnx-max_k)/(N-1-max_k)
    return rnx

if __name__ == "__main__":
    from datasets import load_dataset, load_dataset_knn
    from test_sklearn import get_nne_rate

    K = 5
    rptk = RPTK(K)

    # N = 2048
    # D = 2
    # dataX = np.random.random((N,D)).astype(np.float32)
    
    DATA_SET = "CIFAR"
    dataX, dataY = load_dataset(DATA_SET)

    real_sqd_dist, real_indices = load_dataset_knn(DATA_SET)
    real_indices = real_indices[:,:K]
    real_sqd_dist = real_sqd_dist[:,:K]

    indices, dist = rptk.find_nearest_neighbors(dataX, n_trees=1,verbose=3)

    print(indices, dist)
    print(real_indices, real_sqd_dist)

    print(get_nne_rate(real_indices,indices))

    print(np.sum(indices==-1))