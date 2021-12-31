import numpy as np
from gpu_rsfk.RSFK import RSFK
import faiss
import time
from examples.utils.knn_compare import get_nne_rate, create_recall_eps, KnnResult

np.random.seed(0)

NB = 1000000
NQ = 100000
# NB = 10000
# NQ = 1000
D = 128

t_init = time.time()
points = np.random.random((NQ,D))
query_points = np.random.random((NQ,D))

# points = np.arange(NB*D).reshape(-1, D)
# query_points = np.arange(NQ*D).reshape(-1, D)

points = points.astype(np.float32)
query_points = points.astype(np.float32)

K = 128 # number of neighbors



res = faiss.StandardGpuResources()  # use a single GPU
init_t = time.time()
index = faiss.IndexFlatL2(D)
index = faiss.index_cpu_to_gpu(res, 0, index)
index.train(points)
index.add(points)
D, I = index.search(query_points, K)     # search
t = time.time() - init_t

print("FAISS FLATL2 takes {} seconds".format(t))








rsfk = RSFK(random_state=0)

indices, dist = rsfk.find_nearest_neighbors_ann(
    points,
    query_points,
    K,
    # verbose=2,
    verbose=1,
    n_trees=1,
    add_bit_random_motion=True,
    min_tree_children=(K+1)*3,
    max_tree_children=9*(K+1)) # number of trees
    # min_tree_children=K+1,
    # max_tree_children=3*(K+1)) # number of trees

# print(indices) # the neighborhood of each point
# print(dist) # the squared distance to each neighbor
print(f"{time.time() - t_init}")
nne_rate = get_nne_rate(I,indices, max_k=K)
print(f"NNE Rate: {nne_rate}")
negative_indices = np.sum(indices==-1)
if negative_indices > 0:
    raise Exception('{} Negative indices'.format(negative_indices))


for nprobe in range(5, 101, 5):
    print(f"nprobe={nprobe}")
    t_init = time.time()
    res = faiss.StandardGpuResources()  # use a single GPU
    # Set the base and query points as the same 
    xb = np.require(points, np.float32, ['CONTIGUOUS', 'ALIGNED'])
    xq = np.require(query_points, np.float32, ['CONTIGUOUS', 'ALIGNED'])
    nq, d = xq.shape

    # Parameters that control the trade-off between Quality and Time  
    nlist = int(np.sqrt(10000))
    # nprobe = 20

    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
    index.nprobe = nprobe
    index = faiss.index_cpu_to_gpu(res, 0, index)

    index.train(xb)
    index.add(xb)
    print(f"Index: {time.time() - t_init}")
    t_init = time.time()

    dist, indices = index.search(xq, K)     # actual search

    # print(indices)
    # print(dist)
    print(f"Query: {time.time() - t_init}")
    nne_rate = get_nne_rate(I,indices, max_k=K)
    print(f"NNE Rate: {nne_rate}")