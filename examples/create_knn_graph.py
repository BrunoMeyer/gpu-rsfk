from gpu_rsfk.RSFK import RSFK
try:
    FAISS_LIB_INST = True
    import faiss
except:
    FAISS_LIB_INST = False

import numpy as np
import argparse

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Parameters used to init experiments")
    parser.add_argument("-n", "--n_points", help="Number of points",
                        type=int, default=1000000)
    parser.add_argument("-d", "--n_dim", help="Number of points",
                        type=int, default=100)
    parser.add_argument("-v", "--rsfk_verbose", help="RSFK verbose level",
                        type=int, default=1)
    parser.add_argument("-k", "--num_neigh", help="Number of Neighbors",
                        type=int, default=32)
    

    args = parser.parse_args()

    n_points = args.n_points
    n_dim = args.n_dim
    rsfk_verbose = args.rsfk_verbose
    K = args.num_neigh

    points = np.random.random((n_points,n_dim))


    min_tree_children = 33
    max_tree_children = 128
    n_trees = 50

    print("Number of points {}".format(n_points))
    print("Number of dimensions {}".format(n_dim))

    print("Creating approximated K-NNG with RSFK")
    rsfk = RSFK(random_state=0)
    indices, dist = rsfk.find_nearest_neighbors(points,
                                                K,
                                                n_trees=n_trees)
    print(indices)
    print(dist)

    if FAISS_LIB_INST:
        print("\n\nCreating approximated K-NNG with FAISS")
        res = faiss.StandardGpuResources()  # use a single GPU
        xb = np.require(points, np.float32, ['CONTIGUOUS', 'ALIGNED'])
        xq = np.require(points, np.float32, ['CONTIGUOUS', 'ALIGNED'])
        nq, d = xq.shape

        # Parameters that control the trade-off between Quality and Time  
        nlist = int(np.sqrt(nq))
        nprobe = 20

        quantizer = faiss.IndexFlatL2(d)  # the other index
        index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
        index.nprobe = nprobe              # default nprobe is 1, try a few more
        
        index.train(xb)
        index.add(xb)
        index = faiss.index_cpu_to_gpu(res, 0, index)
        
        dist, indices = index.search(xq, K)     # actual search
        
        print(indices)
        print(dist)