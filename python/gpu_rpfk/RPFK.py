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

class RPFK(object):
    def __init__(self,
                 num_nearest_neighbors,
                 random_state=0,
                 add_bit_random_motion=True,
                 nn_exploring_factor=1
            ):
        self.num_nearest_neighbors = int(num_nearest_neighbors)
        self.random_state = int(random_state)
        self.add_bit_random_motion = bool(add_bit_random_motion)
        self.nn_exploring_factor = int(nn_exploring_factor)
        
        # Build the hooks for the BH T-SNE library
        self._path = pkg_resources.resource_filename('RPFK','') # Load from current location
        # self._faiss_lib = np.ctypeslib.load_library('libfaiss', self._path) # Load the ctypes library
        # self._gpufaiss_lib = np.ctypeslib.load_library('libgpufaiss', self._path) # Load the ctypes library
        self._lib = np.ctypeslib.load_library('librpfk', self._path) # Load the ctypes library
        
        # Hook the BH T-SNE function
        self._lib.pymodule_rpfk_knn.restype = None
        self._lib.pymodule_rpfk_knn.argtypes = [ 
                ctypes.c_int, # number of trees
                ctypes.c_int, # number of nearest neighbors
                ctypes.c_int, # total of points
                ctypes.c_int, # dimensions of points
                ctypes.c_int, # maximum tree node children
                ctypes.c_int, # maximum depth of tree
                ctypes.c_int, # verbose (1,2,3 or 4)
                ctypes.c_int, # random state/seed
                ctypes.c_int, # Nearest Neighbor exploring factor
                np.ctypeslib.ndpointer(np.float32, ndim=2, flags='ALIGNED, CONTIGUOUS'), # points
                np.ctypeslib.ndpointer(np.int32, ndim=2, flags='ALIGNED, CONTIGUOUS, WRITEABLE'), # knn-indices
                np.ctypeslib.ndpointer(np.float32, ndim=2, flags='ALIGNED, CONTIGUOUS, WRITEABLE'), # knn-sqd-distances
                # TODO: run name - Char pointer?
                ]

    def find_nearest_neighbors(self, points, n_trees=1, max_tree_depth=500,
                               verbose=1, max_tree_chlidren=-1,
                               transposed_points=False, random_motion_force=1.0):
        n_trees = int(n_trees)
        max_tree_depth = int(max_tree_depth)
        verbose = int(verbose)
        max_tree_chlidren = int(max_tree_chlidren)
        N = points.shape[0]
        D = points.shape[1]
        K = self.num_nearest_neighbors
        
        if max_tree_chlidren == -1:
            max_tree_chlidren = 2*K+2
        if max_tree_chlidren < 2*K+1:
            raise Exception('max_tree_chlidren = {} \t => max_tree_chlidren must be at least 2*K+1'.format(max_tree_chlidren))
        
        # if(self.add_bit_random_motion):
        #     points=points + np.random.uniform(-0.0001,0.0001,points.shape)

        if(self.add_bit_random_motion):
            for d in range(D):
                min_d = np.min(points[:,d])
                max_d = np.max(points[:,d])
                range_uniform = (max_d - min_d)/N
                points[:,d]=points[:,d] + random_motion_force*np.random.uniform(-range_uniform,range_uniform,N)
        
        if not transposed_points:
            self.points = np.require(points.T, np.float32, ['CONTIGUOUS', 'ALIGNED'])
        else:
            self.points = np.require(points, np.float32, ['CONTIGUOUS', 'ALIGNED'])
        
        # self.points = np.require(points, np.float32, ['CONTIGUOUS', 'ALIGNED'])

        self._knn_indices = np.require(np.full((N,K), -1), np.int32, ['CONTIGUOUS', 'ALIGNED', 'WRITEABLE'])
        self._knn_squared_dist = np.require(np.full((N,K), np.inf), np.float32, ['CONTIGUOUS', 'ALIGNED', 'WRITEABLE'])


        self._lib.pymodule_rpfk_knn(
                ctypes.c_int(n_trees), # number of trees
                ctypes.c_int(K), # number of nearest neighbors
                ctypes.c_int(N), # total of points
                ctypes.c_int(D), # dimensions of points
                ctypes.c_int(max_tree_chlidren), # maximum depth of tree
                ctypes.c_int(max_tree_depth), # maximum depth of tree
                ctypes.c_int(verbose), # verbose (1,2,3 or 4)
                ctypes.c_int(self.random_state), # random state/seed
                ctypes.c_int(self.nn_exploring_factor), # random state/seed
                self.points,
                self._knn_indices,
                self._knn_squared_dist)

        return self._knn_indices, self._knn_squared_dist



def test():
    test_N = 10000
    test_D = 200
    test_K = 30
    dataX = np.random.random((test_N, test_D))
    rptk = RPFK(test_K, random_state=0, nn_exploring_factor=2,
                add_bit_random_motion=True)
    indices, dist = rptk.find_nearest_neighbors(dataX,
                                                max_tree_chlidren=128,
                                                # max_tree_chlidren=len(dataX),
                                                max_tree_depth=5000,
                                                n_trees=10,
                                                transposed_points=True,
                                                random_motion_force=0.01,
                                                # verbose=0)
                                                # verbose=1)
                                                verbose=2)

    negative_indices = np.sum(indices==-1)
    if negative_indices > 0:
        raise Exception('{} Negative indices'.format(negative_indices))

if __name__ == "__main__":
    test()
    exit()