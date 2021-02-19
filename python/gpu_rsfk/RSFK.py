'''
This file is part of the GPU-RSFK Project (https://github.com/BrunoMeyer/gpu-rsfk).

BSD 3-Clause License

Copyright (c) 2021, Bruno Henrique Meyer, Wagner M. Nunan Zola
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''

import numpy as np
import ctypes
import pkg_resources
import time

def random_motion(points, random_motion_force):
    """Move each point with small value

        Parameters
        ----------
        random_state: int
            The seed used to construct the trees. After the construction of each
            tree, this parameters will be increase by 1

        random_motion_force: float
            Magnitude impact of the random motion.
            If it is 0.0, then the points will not be moved
        """
    N,D = points.shape
    # For each dimension
    for d in range(D):
        # Get the minimum and maximum values for each point in this dimension
        min_d = np.min(points[:,d])
        max_d = np.max(points[:,d])

        # Estimate a small value based in the total of points and
        # points position in the dimension
        range_uniform = (max_d - min_d)/N

        # Apply a random motion in the points with the estimated motion value
        # and the force specified by user 
        points[:,d]=points[:,d] + random_motion_force*np.random.uniform(-range_uniform,range_uniform,N)

class ForestLog(object):

    def __init__(self, data):
        attr_names = [
            "Tree Depth",
            "Max Leaf Size",
            "Min Leaf Size",
            "Total Leaves",
            "Tree Initialization Time",
            "Total Tree Building Time",
            "Check Active Points Time",
            "Check Points Side Time",
            "Count New Nodes Time",
            "Dynamic Memory Allocation Time",
            "Preprocessing Split Points Time",
            "Create Nodes Time",
            "Update Nodes Time",
            "Bucket Creation Time",
            "End Tree Time",
            "KNN Time",
        ]
        int_attrs = ["Tree Depth", "Max Leaf Size",
                     "Min Leaf Size", "Total Leaves"]

        self.data = {}
        n_trees = int((len(data)-2)/16)
        for i, attr in enumerate(attr_names):
            values = data[i*n_trees:(i+1)*n_trees]
            if attr in int_attrs:
                values = [int(round(v)) for v in values]
            self.data[attr] = values

class RSFK(object):
    def __init__(self, random_state=0):
        """Initialization method for barnes hut RSFK class.

        Parameters
        ----------
        random_state: int
            The seed used to construct the trees. After the construction of each
            tree, this parameters will be increase by 1
        """
        self.random_state = int(random_state)
        
        self._path = pkg_resources.resource_filename('gpu_rsfk','') # Load from current location
        self._lib = np.ctypeslib.load_library('librsfk', self._path) # Load the ctypes library
        # self._lib = np.ctypeslib.load_library('librsfk', ".") # Load the ctypes library
        
        # Hook the RSFK KNN methods
        self._lib.pymodule_rsfk_knn.restype = None
        self._lib.pymodule_rsfk_knn.argtypes = [ 
            ctypes.c_int, # number of trees
            ctypes.c_int, # number of nearest neighbors
            ctypes.c_int, # total of points
            ctypes.c_int, # dimensions of points
            ctypes.c_int, # minimum tree node children
            ctypes.c_int, # maximum tree node children
            ctypes.c_int, # maximum depth of tree
            ctypes.c_int, # verbose (1,2,or 3)
            ctypes.c_int, # random state/seed
            ctypes.c_int, # Nearest Neighbor exploring factor
            np.ctypeslib.ndpointer(np.float32, ndim=2, flags='ALIGNED, CONTIGUOUS'), # points
            np.ctypeslib.ndpointer(np.int32, ndim=2, flags='ALIGNED, CONTIGUOUS, WRITEABLE'), # knn-indices
            np.ctypeslib.ndpointer(np.float32, ndim=2, flags='ALIGNED, CONTIGUOUS, WRITEABLE'), # knn-sqd-distances
            np.ctypeslib.ndpointer(np.float32, ndim=1, flags='ALIGNED, CONTIGUOUS, WRITEABLE'), # log_forest
            # TODO: run name - Char pointer?
            ]

        self._lib.pymodule_cluster_by_sample_tree.restype = None
        self._lib.pymodule_cluster_by_sample_tree.argtypes = [ 
            ctypes.c_int, # total of points
            ctypes.c_int, # dimensions of points
            ctypes.c_int, # minimum tree node children
            ctypes.c_int, # maximum tree node children
            ctypes.c_int, # maximum depth of tree
            ctypes.c_int, # verbose (1,2,or 3)
            ctypes.c_int, # random state/seed
            np.ctypeslib.ndpointer(np.float32, ndim=2, flags='ALIGNED, CONTIGUOUS'), # points
            ctypes.POINTER(ctypes.POINTER(ctypes.c_int)), # nodes buckets
            ctypes.POINTER(ctypes.POINTER(ctypes.c_int)), # bucket sizes
            ctypes.POINTER(ctypes.c_int), # total leaves
            ctypes.POINTER(ctypes.c_int), # max child
            # TODO: run name - Char pointer?
            ]


        self._lib.pymodule_create_cluster_with_hbgf.restype = None
        self._lib.pymodule_create_cluster_with_hbgf.argtypes = [ 
            ctypes.c_int, # number of trees
            ctypes.c_int, # number of clusters
            ctypes.c_int, # number of eigenvectors
            ctypes.c_int, # total of points
            ctypes.c_int, # dimensions of points
            ctypes.c_int, # minimum tree node children
            ctypes.c_int, # maximum tree node children
            ctypes.c_int, # maximum depth of tree
            ctypes.c_int, # verbose (1,2,or 3)
            ctypes.c_int, # random state/seed
            np.ctypeslib.ndpointer(np.float32, ndim=2, flags='ALIGNED, CONTIGUOUS'), # points
            np.ctypeslib.ndpointer(np.int32, ndim=1, flags='ALIGNED, CONTIGUOUS'), # points
            # TODO: run name - Char pointer?
            ]


        self._lib.pymodule_spectral_clustering_with_knngraph.restype = None
        self._lib.pymodule_spectral_clustering_with_knngraph.argtypes = [ 
            ctypes.c_int, # number of trees
            ctypes.c_int, # number of clusters
            ctypes.c_int, # number of neighbors
            ctypes.c_int, # Nearest Neighbor exploring factor
            ctypes.c_int, # number of eigenvectors
            ctypes.c_int, # total of points
            ctypes.c_int, # dimensions of points
            ctypes.c_int, # minimum tree node children
            ctypes.c_int, # maximum tree node children
            ctypes.c_int, # maximum depth of tree
            ctypes.c_int, # verbose (1,2,or 3)
            ctypes.c_int, # random state/seed
            np.ctypeslib.ndpointer(np.float32, ndim=2, flags='ALIGNED, CONTIGUOUS'), # points
            np.ctypeslib.ndpointer(np.int32, ndim=1, flags='ALIGNED, CONTIGUOUS'), # points
            # TODO: run name - Char pointer?
            ]

    def find_nearest_neighbors(self, points, num_nearest_neighbors,
                               n_trees=-1, max_tree_depth=5000,
                               verbose=0, min_tree_children=-1,
                               max_tree_children=-1,
                               ensure_valid_indices=True,
                               add_bit_random_motion=False,
                               random_motion_force=1.0,
                               nn_exploring_factor=-1,
                               point_in_self_neigh=True):
        """Creation of the K-NNG from a set of points.

        Parameters
        ----------
        points : ndarray of shape (`n_points`, `n_dimensions`)
            The set of `n_points` with points with `n_dimensions` dimensions
        num_nearest_neighbors : int
            The Number of neighbors (K) considered in the K-NN graph
        n_trees : int, optional
            The Number of trees used to construct the K-NN graph
            If not specified, a heuristic will be used to estimate a proper
            number based on empiric experiments
            This heuristic will consider the total of points, dimensions, and
            the bucket size
        max_tree_depth : int, optional
            The depth limit for each tree built
            If this value is to low, it can lead to an error during execution
        verbose : bool, optional
            Verbosity level: 0, 1, 2 or 3
        min_tree_children : int, optional
            The minimum number of points that will be contained in each bucket
        max_tree_children : int, optional
            The maximum number of points that will be contained in each bucket
            It must be at least 2*`min_tree_children`
            It must be at most 1024
        ensure_valid_indices : bool, optional
            If ensure_valid_indices is True and `min_tree_children` < K+1,
            then create an additional tree with `min_tree_children`=`num_nearest_neighbor`+1
            and `max_tree_children`=3*(`num_nearest_neighbor`+1)
            If ensure_valid_indices is False and `min_tree_children` < K+1,
            then it is possible that some points doesnt get the expected
            number of neighbors (will receive -1 index as result instead
            of others points index)
            If ensure_valid_indices is False and `min_tree_children` >= K+1,
            it has none effect
        add_bit_random_motion : bool, optional
            If the points contain points with the same position,
            the construction of a tree reaches the max_tree_depth
            To prevent this, set `add_bit_random_motion`, which automatically will
            estimate a random motion to each point and can be controlled by
            `random_motion_force` parameter
        random_motion_force : float, optional
            Used only if `add_bit_random_motion` is True
            Check `random_motion` function for more details
        nn_exploring_factor : int, optional
            The Number of iterations of the nearest neighborhood exploration step
            If not specified, it will be 1 if `num_nearest_neighbors` < 32,
            or 0 otherwise.
            If it is 0, then the neighborhood exploration will not be used
        point_in_self_neigh : bool, optional
            If True, the index of a point will be considered in its self neighborhood
        
        Returns
        -------
        knn_indices : ndarray of shape (`num_nearest_neighbors`, `n_dimensions`)
            The matrix with the neighborhood of each point (index of points)
            If `ensure_valid_indices`=False, some indexes may be set as -1
        knn_squared_dist : ndarray of shape (`num_nearest_neighbors`, `n_dimensions`)
            For each neighbor in `knn_indices`, the `knn_squared_dist`
            contains the squared distance computed
        """
        points = np.require(points, np.float32, ['CONTIGUOUS', 'ALIGNED'])

        max_tree_depth = int(max_tree_depth)
        verbose = int(verbose)
        max_tree_children = int(max_tree_children)
        N = points.shape[0]
        D = points.shape[1]
        K = num_nearest_neighbors
        n_trees = int(n_trees)

        if nn_exploring_factor == -1:
            if K <= 32:
                nn_exploring_factor = 1
            else:
                nn_exploring_factor = 0

        if min_tree_children == -1:
            min_tree_children = K+1
        if min_tree_children < 0:
            raise Exception('min_tree_children = {} \t => min_tree_children must be at least 1'.format(min_tree_children))
        if max_tree_children == -1:
            max_tree_children = 2*min_tree_children
        if max_tree_children < 2*min_tree_children:
            raise Exception('min_tree_children = {}, max_tree_children = {} \t => max_tree_children must be at least 2*min_tree_children'.format(min_tree_children, max_tree_children))
        
        if n_trees == -1:
            avgBucketSize = (min_tree_children+max_tree_children)/2
            n_trees = (N*D)/(avgBucketSize*24913) # Magic number collected by experiments (reported in paper)
            n_trees = max(np.ceil(n_trees), 50)
            if verbose > 0:
                print("Automatically setting n_trees to {}".format(n_trees))
            
        n_trees = int(n_trees)
        if(add_bit_random_motion):
            random_motion(points, random_motion_force)

        # TODO: Prevent np.require to create a copy of the data. This double the memory usage              
        points = np.require(points, np.float32, ['CONTIGUOUS', 'ALIGNED'])
        
        init_knn_indices = np.full((N,K), -1)
        if point_in_self_neigh:
            init_knn_indices[:,0] = np.arange(N)
        
        init_squared_dist = np.full((N,K), np.inf)
        if point_in_self_neigh:
            init_squared_dist[:,0] = np.zeros(N)

        knn_indices = np.require(init_knn_indices, np.int32, ['CONTIGUOUS', 'ALIGNED', 'WRITEABLE'])
        knn_squared_dist = np.require(init_squared_dist, np.float32, ['CONTIGUOUS', 'ALIGNED', 'WRITEABLE'])

        t_init = time.time()
        
        log_forest = np.require(np.zeros(n_trees*16+2), np.float32, ['CONTIGUOUS', 'ALIGNED', 'WRITEABLE'])
        self._lib.pymodule_rsfk_knn(
                ctypes.c_int(n_trees), # number of trees
                ctypes.c_int(K), # number of nearest neighbors
                ctypes.c_int(N), # total of points
                ctypes.c_int(D), # dimensions of points
                ctypes.c_int(min_tree_children), # minimum tree node children
                ctypes.c_int(max_tree_children), # maximum tree node children
                ctypes.c_int(max_tree_depth), # maximum depth of tree
                ctypes.c_int(verbose), # verbose (1,2,or 3)
                ctypes.c_int(self.random_state), # random state/seed
                ctypes.c_int(nn_exploring_factor), # random state/seed
                points,
                knn_indices,
                knn_squared_dist,
                log_forest)

        self.log_forest = ForestLog(log_forest)

        if ensure_valid_indices and min_tree_children < K+1:
            self._lib.pymodule_rsfk_knn(
                    ctypes.c_int(1), # number of trees
                    ctypes.c_int(K), # number of nearest neighbors
                    ctypes.c_int(N), # total of points
                    ctypes.c_int(D), # dimensions of points
                    ctypes.c_int(K+1), # minimum tree node children
                    ctypes.c_int(3*(K+1)), # maximum tree node children
                    ctypes.c_int(max_tree_depth), # maximum depth of tree
                    ctypes.c_int(verbose), # verbose (1,2,or 3)
                    ctypes.c_int(self.random_state), # random state/seed
                    ctypes.c_int(nn_exploring_factor), # random state/seed
                    points,
                    knn_indices,
                    knn_squared_dist)
        
        self._last_search_time = time.time() - t_init
        return knn_indices, knn_squared_dist

    def cluster_by_sample_tree(self, points, max_tree_depth=5000,
                               verbose=1, min_tree_children=32,
                               max_tree_children=128,
                               add_bit_random_motion=False,
                               random_motion_force=1.0,
                               numpy_result_format=True):
        """Creates a partition of the points with only one tree.

        Parameters
        ----------
        points : ndarray of shape (`n_points`, `n_dimensions`)
            The set of `n_points` with points with `n_dimensions` dimensions
        max_tree_depth : int, optional
            The depth limit for each tree built
            If this value is to low, it can lead to an error during execution
        verbose : bool, optional
            Verbosity level: 0, 1, 2 or 3
        min_tree_children : int, optional
            The minimum number of points that will be contained in each bucket
        max_tree_children : int, optional
            The maximum number of points that will be contained in each bucket
            It must be at least 2*`min_tree_children`
            It must be at most 1024
        add_bit_random_motion : bool, optional
            If the points contain points with the same position,
            the construction of a tree reaches the max_tree_depth
            To prevent this, set `add_bit_random_motion`, which automatically will
            estimate a random motion to each point and can be controlled by
            `random_motion_force` parameter
        random_motion_force : float, optional
            Used only if `add_bit_random_motion` is True
            Check `random_motion` function for more details
        numpy_result_format : bool, optional
            If True, convert the result to numpy ndarrays
            If False, the return results will be structured with ctype classes
        
        Returns
        -------
        total_leaves : int or ctypes.c_int
            The number of created buckets
        max_child : int or ctypes.c_int
            The maximum number of points inside a bucket
        nodes_buckets : LP_c_int or ndarray of shape (`total_leaves`*`max_child`,)
            The points indexes inside each bucket
            Each bucket contains `max_child` elements in the array
            In the buckets with less than `max_child` elements, the gaps
            will be filled with -1
            Buckets are sequenced in `nodes_buckets` array with sequences
            with size `max_child` 
        bucket_sizes : LP_c_int or ndarray of shape (`total_leaves`,)
            The size of each bucket
        """
        points = np.require(points, np.float32, ['CONTIGUOUS', 'ALIGNED'])

        max_tree_depth = int(max_tree_depth)
        verbose = int(verbose)
        max_tree_children = int(max_tree_children)
        N = points.shape[0]
        D = points.shape[1]
        K = -1 # Only KNN update with buckets need a number of neighbors
        
        if min_tree_children < 0:
            raise Exception('min_tree_children = {} \t => min_tree_children must be at least 1'.format(min_tree_children))
        if max_tree_children < 2*min_tree_children:
            raise Exception('min_tree_children = {}, max_tree_children = {} \t => max_tree_children must be at least 2*min_tree_children'.format(min_tree_children, max_tree_children))

        
        if(add_bit_random_motion):
            random_motion(points, random_motion_force)

        t_init = time.time()
        
        nodes_buckets = ctypes.POINTER(ctypes.c_int)()
        bucket_sizes  = ctypes.POINTER(ctypes.c_int)()
        total_leaves = ctypes.c_int(0)
        max_child = ctypes.c_int(0)
        

        self._lib.pymodule_cluster_by_sample_tree(
                ctypes.c_int(N), # total of points
                ctypes.c_int(D), # dimensions of points
                ctypes.c_int(min_tree_children), # minimum tree node children
                ctypes.c_int(max_tree_children), # maximum tree node children
                ctypes.c_int(max_tree_depth), # maximum depth of tree
                ctypes.c_int(verbose), # verbose (1,2,or 3)
                ctypes.c_int(self.random_state), # random state/seed
                points,
                ctypes.byref(nodes_buckets),
                ctypes.byref(bucket_sizes),
                ctypes.byref(total_leaves),
                ctypes.byref(max_child))

        
        self._last_cluster_time = time.time() - t_init

        if numpy_result_format:
            total_leaves = np.int(total_leaves.value)
            max_child = np.int(max_child.value)
            nodes_buckets = np.ctypeslib.as_array(nodes_buckets,
                                                  shape=(1,total_leaves*max_child))[0]
            bucket_sizes = np.ctypeslib.as_array(bucket_sizes,
                                                 shape=(1,total_leaves))[0]
        
        return total_leaves, max_child, nodes_buckets, bucket_sizes

    def create_cluster_with_hbgf(self, points, n_clusters,
                                 n_trees=10, n_eig_vects=2,
                                 max_tree_depth=5000,
                                 verbose=1, min_tree_children=32,
                                 max_tree_children=128,
                                 add_bit_random_motion=False,
                                 random_motion_force=1.0):
        """Creates a partition of the points with several random sample trees
           The partitions of each tree are ensembled with HBGF algorithm
           using Spectral Clustering (implemented in nvGRAPH lib)

        Parameters
        ----------
        points : ndarray of shape (`n_points`, `n_dimensions`)
            The set of `n_points` with points with `n_dimensions` dimensions
        n_clusters : int
            Number of clusters
        n_trees : int, optional
            Number of trees
        n_eig_vects : int, optional
            Number of eigenvectors used in Spectral Clustering algorithm.
            Must be less or equal than n_clusters and len(points) 
        max_tree_depth : int, optional
            The depth limit for each tree built
            If this value is to low, it can lead to an error during execution
        verbose : bool, optional
            Verbosity level: 0, 1, 2 or 3
        min_tree_children : int, optional
            The minimum number of points that will be contained in each bucket
        max_tree_children : int, optional
            The maximum number of points that will be contained in each bucket
            It must be at least 2*`min_tree_children`
            It must be at most 1024
        add_bit_random_motion : bool, optional
            If the points contain points with the same position,
            the construction of a tree reaches the max_tree_depth
            To prevent this, set `add_bit_random_motion`, which automatically will
            estimate a random motion to each point and can be controlled by
            `random_motion_force` parameter
        random_motion_force : float, optional
            Used only if `add_bit_random_motion` is True
            Check `random_motion` function for more details
        
        Returns
        -------
        cluster : int ndarray of shape (`len(points)`,)
            The cluster index of each point
        """
        points = np.require(points, np.float32, ['CONTIGUOUS', 'ALIGNED'])
        cluster = np.full(len(points), -1, dtype=np.int32)
        cluster = np.require(cluster, np.int32, ['CONTIGUOUS', 'ALIGNED'])
        
        max_tree_depth = int(max_tree_depth)
        verbose = int(verbose)
        max_tree_children = int(max_tree_children)
        N = points.shape[0]
        D = points.shape[1]
        K = -1 # Only KNN update with buckets need a number of neighbors
        
        if min_tree_children < 0:
            raise Exception('min_tree_children = {} \t => min_tree_children must be at least 1'.format(min_tree_children))
        if max_tree_children < 2*min_tree_children:
            raise Exception('min_tree_children = {}, max_tree_children = {} \t => max_tree_children must be at least 2*min_tree_children'.format(min_tree_children, max_tree_children))

        
        if(add_bit_random_motion):
            random_motion(points, random_motion_force)

        t_init = time.time()
        
        self._lib.pymodule_create_cluster_with_hbgf(
            ctypes.c_int(n_trees), # number of trees
            ctypes.c_int(n_clusters), # number of clusters
            ctypes.c_int(n_eig_vects), # number of eigenvectors
            ctypes.c_int(N), # total of points
            ctypes.c_int(D), # dimensions of points
            ctypes.c_int(min_tree_children), # minimum tree node children
            ctypes.c_int(max_tree_children), # maximum tree node children
            ctypes.c_int(max_tree_depth), # maximum depth of tree
            ctypes.c_int(verbose), # verbose (1,2,or 3)
            ctypes.c_int(self.random_state), # random state/seed
            points,
            cluster)

        
        self._last_cluster_time = time.time() - t_init
        
        return cluster

        
    def spectral_clustering_with_knngraph(self, points, n_clusters,
                                          num_nearest_neighbors=32,
                                          n_trees=10, n_eig_vects=2,
                                          max_tree_depth=5000,
                                          verbose=1, min_tree_children=32,
                                          max_tree_children=128,
                                          add_bit_random_motion=False,
                                          random_motion_force=1.0,
                                          nn_exploring_factor=-1):
        """Creates a partition of the points with a knn-graph
           The K-NN graph is used to execute the Spectral Clustering algorithm
           of nvGRAPH lib

        Parameters
        ----------
        points : ndarray of shape (`n_points`, `n_dimensions`)
            The set of `n_points` with points with `n_dimensions` dimensions
        n_clusters : int
            Number of clusters
        num_nearest_neighbors : int, optional
            Number of neighbors in the k-nn graph
        n_trees : int, optional
            Number of trees
        n_eig_vects : int, optional
            Number of eigenvectors used in Spectral Clustering algorithm.
            Must be less or equal than n_clusters and len(points) 
        max_tree_depth : int, optional
            The depth limit for each tree built
            If this value is to low, it can lead to an error during execution
        verbose : bool, optional
            Verbosity level: 0, 1, 2 or 3
        min_tree_children : int, optional
            The minimum number of points that will be contained in each bucket
        max_tree_children : int, optional
            The maximum number of points that will be contained in each bucket
            It must be at least 2*`min_tree_children`
            It must be at most 1024
        add_bit_random_motion : bool, optional
            If the points contain points with the same position,
            the construction of a tree reaches the max_tree_depth
            To prevent this, set `add_bit_random_motion`, which automatically will
            estimate a random motion to each point and can be controlled by
            `random_motion_force` parameter
        random_motion_force : float, optional
            Used only if `add_bit_random_motion` is True
            Check `random_motion` function for more details
        nn_exploring_factor : int, optional
            The Number of iterations of the nearest neighborhood exploration step
            If not specified, it will be 1 if `num_nearest_neighbors` < 32,
            or 0 otherwise.
            If it is 0, then the neighborhood exploration will not be used
        
        Returns
        -------
        cluster : int ndarray of shape (`len(points)`,)
            The cluster index of each point
        """
        points = np.require(points, np.float32, ['CONTIGUOUS', 'ALIGNED'])
        cluster = np.full(len(points), -1, dtype=np.int32)
        cluster = np.require(cluster, np.int32, ['CONTIGUOUS', 'ALIGNED'])
        
        max_tree_depth = int(max_tree_depth)
        verbose = int(verbose)
        max_tree_children = int(max_tree_children)
        N = points.shape[0]
        D = points.shape[1]
        K = -1 # Only KNN update with buckets need a number of neighbors
        
        if min_tree_children < 0:
            raise Exception('min_tree_children = {} \t => min_tree_children must be at least 1'.format(min_tree_children))
        if max_tree_children < 2*min_tree_children:
            raise Exception('min_tree_children = {}, max_tree_children = {} \t => max_tree_children must be at least 2*min_tree_children'.format(min_tree_children, max_tree_children))

        if nn_exploring_factor == -1:
            if K <= 32:
                nn_exploring_factor = 1
            else:
                nn_exploring_factor = 0

        if(add_bit_random_motion):
            random_motion(points, random_motion_force)

        t_init = time.time()
        
        self._lib.pymodule_spectral_clustering_with_knngraph(
            ctypes.c_int(n_trees), # number of trees
            ctypes.c_int(n_clusters), # number of clusters
            ctypes.c_int(num_nearest_neighbors), # number of neighbors
            ctypes.c_int(nn_exploring_factor), # number of neighbors
            ctypes.c_int(n_eig_vects), # number of eigenvectors
            ctypes.c_int(N), # total of points
            ctypes.c_int(D), # dimensions of points
            ctypes.c_int(min_tree_children), # minimum tree node children
            ctypes.c_int(max_tree_children), # maximum tree node children
            ctypes.c_int(max_tree_depth), # maximum depth of tree
            ctypes.c_int(verbose), # verbose (1,2,or 3)
            ctypes.c_int(self.random_state), # random state/seed
            points,
            cluster)

        
        self._last_cluster_time = time.time() - t_init
        
        return cluster

def test1():
    test_N = 3000
    test_D = 200
    test_K = 30
    dataX = np.random.random((test_N, test_D))
    rptk = RSFK(random_state=0)
    indices, dist = rptk.find_nearest_neighbors(dataX,
                                                test_K,
                                                max_tree_children=128,
                                                # max_tree_children=len(dataX),
                                                max_tree_depth=5000,
                                                n_trees=10,
                                                random_motion_force=0.01,
                                                nn_exploring_factor=2,
                                                add_bit_random_motion=True,
                                                # verbose=0)
                                                # verbose=1)
                                                verbose=2)

    negative_indices = np.sum(indices==-1)
    if negative_indices > 0:
        raise Exception('{} Negative indices'.format(negative_indices))

def test2():
    n_points = 2**12
    n_dim = 99
    rsfk_verbose = 2
    points = np.random.random((n_points,n_dim))
    min_tree_children = int(n_points/30)
    max_tree_children = 3*min_tree_children
    print("Number of points {}".format(n_points))
    print("Number of dimensions {}".format(n_dim))

    rsfk = RSFK(random_state=0)

    result = rsfk.cluster_by_sample_tree(points,
                                         min_tree_children=min_tree_children,
                                         max_tree_children=max_tree_children,
                                         max_tree_depth=5000,
                                         random_motion_force=0.1,
                                         verbose=rsfk_verbose,
                                         add_bit_random_motion=False)

    total_leaves, max_child, nodes_buckets, bucket_sizes = result
    
    
    # result = np.full(len(points), -1, dtype=np.int32)
    # offset = 0
    # while offset < len(nodes_buckets):
    #     bucket_slice = nodes_buckets[offset:offset+max_child]
    #     # -1 values represent empty slots on buckets
    #     bucket_slice = bucket_slice[np.where(bucket_slice != -1)]
    #     result[bucket_slice] = int(offset/max_child)
    #     offset+=max_child
    # result = np.array(result)
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(figsize=(4,4))
    # for y in set(list(result)):
    #     ax.scatter(points[result==y,0], points[result==y,1], label=y,
    #                alpha=0.4, s=10, cmap='tab10')
    # plt.show()

    print("Total leaves: {}".format(total_leaves))

def test3():
    n_points = 2**12
    n_dim = 99
    n_points = 1000
    n_dim = 2
    points = np.random.random((n_points,n_dim))

    from sklearn import datasets
    n_points = 1024
    n_dim = 2


    noisy_circles = datasets.make_circles(n_samples=n_points, factor=.5,
                                        noise=.05)
    noisy_moons = datasets.make_moons(n_samples=n_points, noise=.05)
    blobs = datasets.make_blobs(n_samples=n_points, random_state=8,
                                cluster_std=0.5, n_features=n_dim)
    no_structure = np.random.rand(n_points, 2), None

    # Anisotropicly distributed data
    random_state = 170
    X, y = datasets.make_blobs(n_samples=n_points, random_state=random_state)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X_aniso = np.dot(X, transformation)
    
    aniso = (X_aniso, y)

    # blobs with varied variances
    varied = datasets.make_blobs(n_samples=n_points,
                                cluster_std=[1.0, 2.5, 0.5],
                                random_state=random_state)

    # n_clusters = 2
    # points, _ = noisy_circles
    
    # n_clusters = 3
    # points, _ = blobs
    # points, _ = aniso
    # points, _ = varied
    
    # points, _ = no_structure
    
    
    datasets = [
        (noisy_circles, 2),
        (noisy_moons, 2),
        (varied, 3),
        (aniso, 3),
        (blobs, 3),
        (no_structure, 3)
        ]
    
    num_nearest_neighbors = 64
    min_tree_children = num_nearest_neighbors+1
    max_tree_children = 3*min_tree_children
    rsfk_verbose = 2
    for dataset, n_clusters in datasets:
        points, _ = dataset
        # min_tree_children = int(n_points/(n_clusters*12))
        # max_tree_children = 3*min_tree_children

        print("Number of points {}".format(n_points))
        print("Number of dimensions {}".format(n_dim))

        rsfk = RSFK(random_state=0)

        result = rsfk.create_cluster_with_hbgf(points, n_clusters,
                                               n_trees=20, n_eig_vects=1,
                                               min_tree_children=min_tree_children,
                                               max_tree_children=max_tree_children,
                                               max_tree_depth=5000,
                                               random_motion_force=0.0,
                                               verbose=rsfk_verbose,
                                               add_bit_random_motion=False)

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(4,4))
        for y in set(list(result)):
            ax.scatter(points[result==y,0], points[result==y,1], label=y,
                    alpha=0.4, s=10, cmap='tab10')
        plt.show()

        print("Cluster result: {}".format(result))

def test4():
    from sklearn import datasets
    n_points = 1024000
    n_dim = 2

    noisy_circles = datasets.make_circles(n_samples=n_points, factor=.5,
                                        noise=.05)
    noisy_moons = datasets.make_moons(n_samples=n_points, noise=.05)
    blobs = datasets.make_blobs(n_samples=n_points, random_state=8,
                                cluster_std=0.5, n_features=n_dim)
    no_structure = np.random.rand(n_points, 2), None

    # Anisotropicly distributed data
    random_state = 170
    X, y = datasets.make_blobs(n_samples=n_points, random_state=random_state)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X_aniso = np.dot(X, transformation)
    
    aniso = (X_aniso, y)

    # blobs with varied variances
    varied = datasets.make_blobs(n_samples=n_points,
                                cluster_std=[1.0, 2.5, 0.5],
                                random_state=random_state)

    # n_clusters = 2
    # points, _ = noisy_circles
    
    # n_clusters = 3
    # points, _ = blobs
    # points, _ = aniso
    # points, _ = varied
    
    # points, _ = no_structure
    
    
    datasets = [
        (noisy_circles, 2),
        (noisy_moons, 2),
        (varied, 3),
        (aniso, 3),
        (blobs, 3),
        (no_structure, 4)
        ]
    
    num_nearest_neighbors = 8
    # min_tree_children = int(n_points/(n_clusters*12))
    # max_tree_children = 3*min_tree_children
    min_tree_children = num_nearest_neighbors+1
    max_tree_children = 3*min_tree_children
    rsfk_verbose = 2
    for dataset, n_clusters in datasets:
        points, _ = dataset

        print("Number of points {}".format(n_points))
        print("Number of dimensions {}".format(n_dim))

        rsfk = RSFK(random_state=0)

        result = rsfk.spectral_clustering_with_knngraph(
            points, n_clusters,
            num_nearest_neighbors=num_nearest_neighbors,
            n_trees=10, n_eig_vects=n_clusters,
            nn_exploring_factor=0,
            min_tree_children=min_tree_children,
            max_tree_children=max_tree_children,
            max_tree_depth=5000,
            random_motion_force=0.01,
            verbose=rsfk_verbose,
            add_bit_random_motion=True)

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(4,4))
        for y in set(list(result)):
            ax.scatter(points[result==y,0], points[result==y,1], label=y,
                    alpha=0.4, s=10, cmap='tab10')
        plt.show()

        print("Cluster result: {}".format(result))

def test5():
    from sklearn import datasets
    n_points = 1024000
    n_dim = 2


    noisy_circles = datasets.make_circles(n_samples=n_points, factor=.5,
                                        noise=.05)
    noisy_moons = datasets.make_moons(n_samples=n_points, noise=.05)
    blobs = datasets.make_blobs(n_samples=n_points, random_state=8,
                                cluster_std=0.5, n_features=n_dim)
    no_structure = np.random.rand(n_points, 2), None

    # Anisotropicly distributed data
    random_state = 170
    X, y = datasets.make_blobs(n_samples=n_points, random_state=random_state)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X_aniso = np.dot(X, transformation)
    
    aniso = (X_aniso, y)

    # blobs with varied variances
    varied = datasets.make_blobs(n_samples=n_points,
                                 cluster_std=[1.0, 2.5, 0.5],
                                 random_state=random_state)

    # n_clusters = 2
    # points, _ = noisy_circles
    
    # n_clusters = 3
    # points, _ = blobs
    # points, _ = aniso
    # points, _ = varied
    
    # points, _ = no_structure
    
    
    datasets = [
        (noisy_circles, 2),
        (noisy_moons, 2),
        (varied, 3),
        (aniso, 3),
        (blobs, 3),
        (no_structure, 4)
        ]
    
    num_nearest_neighbors = 8
    min_tree_children = num_nearest_neighbors+1
    max_tree_children = 3*min_tree_children
    rsfk_verbose = 2
    for dataset, n_clusters in datasets:
        points, _ = dataset

        rptk = RSFK(random_state=0)
        indices, dist = rptk.find_nearest_neighbors(points,
                                                    num_nearest_neighbors,
                                                    min_tree_children=min_tree_children,
                                                    max_tree_children=max_tree_children,
                                                    # max_tree_children=len(dataX),
                                                    max_tree_depth=5000,
                                                    n_trees=10,
                                                    random_motion_force=0.01,
                                                    nn_exploring_factor=0,
                                                    add_bit_random_motion=True,
                                                    # verbose=0)
                                                    # verbose=1)
                                                    verbose=2,
                                                    point_in_self_neigh=False)
        
        from scipy.sparse import csr_matrix
        graph_as_knnindices = indices
        indptr = np.arange(
            0,
            graph_as_knnindices.size+graph_as_knnindices.shape[1],
            graph_as_knnindices.shape[1])
        indices = graph_as_knnindices.flatten()
        graph_edge_val = np.ones(indices.size)
        graph_shape = (len(points), len(points))
        csr_graph = csr_matrix((graph_edge_val,indices,indptr),
                               shape=graph_shape)

        from sklearn.cluster import SpectralClustering
        clustering = SpectralClustering(n_clusters=n_clusters,
            # assign_labels="discretize",
            affinity="precomputed",
            # affinity="precomputed_nearest_neighbors",
            # affinity="nearest_neighbors",
            n_neighbors=num_nearest_neighbors,
            # eigen_solver='arpack',
            eigen_solver='amg',
            random_state=0).fit_predict(csr_graph)
            # random_state=0).fit_predict(points)

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(4,4))
        for y in set(list(clustering)):
            ax.scatter(points[clustering==y,0], points[clustering==y,1], label=y,
                    alpha=0.4, s=10, cmap='tab10')
        plt.show()

        # print("Cluster result: {}".format(result))

def test():
    try:
        test1()
        print("Test 1: OK")
    except ValueError:
        print("Test 1: failed!")
        print(ValueError)
        exit(1)
    try:
        test2()
        print("Test 2: OK")
    except ValueError:
        print("Test 2: failed!")
        print(ValueError)
        exit(1)

    # TODO: Implement and fix Spectral Clustering
    # try:
    #     test3()
    #     print("Test 3: OK")
    # except ValueError:
    #     print("Test 3: failed!")
    #     print(ValueError)
    #     exit(1)
    # try:
    #     test4()
    #     print("Test 4: OK")
    # except ValueError:
    #     print("Test 4: failed!")
    #     print(ValueError)
    #     exit(1)
    # try:
    #     test5()
    #     print("Test 4: OK")
    # except ValueError:
    #     print("Test 4: failed!")
    #     print(ValueError)
    #     exit(1)

    print("2/2 tests completed successfully")


if __name__ == "__main__":
    test()
    exit()