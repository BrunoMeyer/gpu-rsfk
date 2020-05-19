import numpy as np
import ctypes
import pkg_resources
import time



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
        self._path = pkg_resources.resource_filename('gpu_rpfk','') # Load from current location
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
                ctypes.c_int, # minimum tree node children
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

        self._lib.pymodule_cluster_by_sample_tree.restype = None
        self._lib.pymodule_cluster_by_sample_tree.argtypes = [ 
                ctypes.c_int, # number of nearest neighbors
                ctypes.c_int, # total of points
                ctypes.c_int, # dimensions of points
                ctypes.c_int, # minimum tree node children
                ctypes.c_int, # maximum tree node children
                ctypes.c_int, # maximum depth of tree
                ctypes.c_int, # verbose (1,2,3 or 4)
                ctypes.c_int, # random state/seed
                np.ctypeslib.ndpointer(np.float32, ndim=2, flags='ALIGNED, CONTIGUOUS'), # points
                ctypes.POINTER(ctypes.POINTER(ctypes.c_int)), # nodes buckets
                ctypes.POINTER(ctypes.POINTER(ctypes.c_int)), # bucket sizes
                ctypes.POINTER(ctypes.c_int), # total leaves
                ctypes.POINTER(ctypes.c_int), # max child
                # TODO: run name - Char pointer?
                ]

    def find_nearest_neighbors(self, points, n_trees=1, max_tree_depth=500,
                               verbose=1, min_tree_children=-1,
                               max_tree_children=-1, transposed_points=False,
                               random_motion_force=1.0,
                               ensure_valid_indices=True):

        n_trees = int(n_trees)
        max_tree_depth = int(max_tree_depth)
        verbose = int(verbose)
        max_tree_children = int(max_tree_children)
        N = points.shape[0]
        D = points.shape[1]
        K = self.num_nearest_neighbors
        
        if min_tree_children == -1:
            min_tree_children = K+1
        if min_tree_children < 0:
            raise Exception('min_tree_children = {} \t => min_tree_children must be at least 1'.format(min_tree_children))
        if max_tree_children == -1:
            max_tree_children = 2*min_tree_children
        if max_tree_children < 2*min_tree_children:
            raise Exception('min_tree_children = {}, max_tree_children = {} \t => max_tree_children must be at least 2*min_tree_children'.format(min_tree_children, max_tree_children))
        
        # if max_tree_children == -1:
        #     max_tree_children = K+1
        # if max_tree_children < K+1:
        #     raise Exception('max_tree_children = {} \t => max_tree_children must be at least K+1'.format(max_tree_children))
        
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

        init_knn_indices = np.full((N,K), -1)
        init_knn_indices[:,0] = np.arange(N)
        init_squared_dist = np.full((N,K), np.inf)
        init_squared_dist[:,0] = np.zeros(N)

        self._knn_indices = np.require(init_knn_indices, np.int32, ['CONTIGUOUS', 'ALIGNED', 'WRITEABLE'])
        self._knn_squared_dist = np.require(init_squared_dist, np.float32, ['CONTIGUOUS', 'ALIGNED', 'WRITEABLE'])

        t_init = time.time()
        # self._lib.pymodule_rpfk_knn(
        #         ctypes.c_int(10), # number of trees
        #         ctypes.c_int(K), # number of nearest neighbors
        #         ctypes.c_int(N), # total of points
        #         ctypes.c_int(D), # dimensions of points
        #         ctypes.c_int(min_tree_children), # minimum tree node children
        #         ctypes.c_int(max_tree_children), # maximum tree node children
        #         ctypes.c_int(max_tree_depth), # maximum depth of tree
        #         ctypes.c_int(verbose), # verbose (1,2,3 or 4)
        #         ctypes.c_int(self.random_state), # random state/seed
        #         ctypes.c_int(self.nn_exploring_factor), # random state/seed
        #         self.points,
        #         self._knn_indices,
        #         self._knn_squared_dist)
        
        self._lib.pymodule_rpfk_knn(
                ctypes.c_int(n_trees), # number of trees
                ctypes.c_int(K), # number of nearest neighbors
                ctypes.c_int(N), # total of points
                ctypes.c_int(D), # dimensions of points
                ctypes.c_int(min_tree_children), # minimum tree node children
                ctypes.c_int(max_tree_children), # maximum tree node children
                ctypes.c_int(max_tree_depth), # maximum depth of tree
                ctypes.c_int(verbose), # verbose (1,2,3 or 4)
                ctypes.c_int(self.random_state), # random state/seed
                ctypes.c_int(self.nn_exploring_factor), # random state/seed
                self.points,
                self._knn_indices,
                self._knn_squared_dist)

        if ensure_valid_indices and min_tree_children < K+1:
            self._lib.pymodule_rpfk_knn(
                    ctypes.c_int(1), # number of trees
                    ctypes.c_int(K), # number of nearest neighbors
                    ctypes.c_int(N), # total of points
                    ctypes.c_int(D), # dimensions of points
                    ctypes.c_int(K+1), # minimum tree node children
                    ctypes.c_int(2*(K+1)), # maximum tree node children
                    ctypes.c_int(max_tree_depth), # maximum depth of tree
                    ctypes.c_int(verbose), # verbose (1,2,3 or 4)
                    ctypes.c_int(self.random_state), # random state/seed
                    ctypes.c_int(self.nn_exploring_factor), # random state/seed
                    self.points,
                    self._knn_indices,
                    self._knn_squared_dist)
        
        self._last_search_time = time.time() - t_init
        return self._knn_indices, self._knn_squared_dist

    def cluster_by_sample_tree(self, points, max_tree_depth=500,
                               verbose=1, min_tree_children=-1,
                               max_tree_children=-1, transposed_points=False,
                               random_motion_force=1.0,
                               ensure_valid_indices=True):

            max_tree_depth = int(max_tree_depth)
            verbose = int(verbose)
            max_tree_children = int(max_tree_children)
            N = points.shape[0]
            D = points.shape[1]
            K = self.num_nearest_neighbors
            
            if min_tree_children == -1:
                min_tree_children = K+1
            if min_tree_children < 0:
                raise Exception('min_tree_children = {} \t => min_tree_children must be at least 1'.format(min_tree_children))
            if max_tree_children == -1:
                max_tree_children = 2*min_tree_children
            if max_tree_children < 2*min_tree_children:
                raise Exception('min_tree_children = {}, max_tree_children = {} \t => max_tree_children must be at least 2*min_tree_children'.format(min_tree_children, max_tree_children))

            
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

            init_knn_indices = np.full((N,K), -1)
            init_knn_indices[:,0] = np.arange(N)
            init_squared_dist = np.full((N,K), np.inf)
            init_squared_dist[:,0] = np.zeros(N)


            t_init = time.time()
            
            nodes_buckets = ctypes.POINTER(ctypes.c_int)()
            bucket_sizes  = ctypes.POINTER(ctypes.c_int)()
            total_leaves = ctypes.c_int(0)
            max_child = ctypes.c_int(0)

            self._lib.pymodule_cluster_by_sample_tree(
                    ctypes.c_int(K), # number of nearest neighbors
                    ctypes.c_int(N), # total of points
                    ctypes.c_int(D), # dimensions of points
                    ctypes.c_int(min_tree_children), # minimum tree node children
                    ctypes.c_int(max_tree_children), # maximum tree node children
                    ctypes.c_int(max_tree_depth), # maximum depth of tree
                    ctypes.c_int(verbose), # verbose (1,2,3 or 4)
                    ctypes.c_int(self.random_state), # random state/seed
                    self.points,
                    ctypes.byref(nodes_buckets),
                    ctypes.byref(bucket_sizes),
                    ctypes.byref(total_leaves),
                    ctypes.byref(max_child))

            
            self._last_search_time = time.time() - t_init
            return total_leaves, max_child, nodes_buckets, bucket_sizes

def test():
    test_N = 3000
    test_D = 200
    test_K = 30
    dataX = np.random.random((test_N, test_D))
    rptk = RPFK(test_K, random_state=0, nn_exploring_factor=2,
                add_bit_random_motion=True)
    indices, dist = rptk.find_nearest_neighbors(dataX,
                                                max_tree_children=128,
                                                # max_tree_children=len(dataX),
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