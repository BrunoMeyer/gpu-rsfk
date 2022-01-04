import numpy as np
from gpu_rsfk.RSFK import RSFK
import faiss
import time
from examples.utils.knn_compare import get_nne_rate, create_recall_eps, KnnResult
from examples.datasets import load_dataset, load_dataset_knn, get_dataset_options

from sklearn.model_selection import train_test_split

# print(get_dataset_options())
# exit()

np.random.seed(0)

'''
NB = 1000000
NQ = 100000
# NB = 10000
# NQ = 1000
D = 1024

t_init = time.time()
points = np.random.random((NQ,D))
query_points = np.random.random((NQ,D))
'''

# dataset = "MNIST"
# points, points_label, points_query, points_query_label = load_dataset(
#     dataset,
#     return_test_data=True)

dataset = "GOOGLE_NEWS300"
dataX, dataY = load_dataset(dataset)

np.random.shuffle(dataX)

size = 1000000
dataX = dataX[:size]
dataY = dataY[:size]

points, points_query, points_label, points_query_label = train_test_split(
    dataX, dataY, test_size=0.10, random_state=42)




NB = len(points)
NQ = len(points_query)
D = points.shape[1]

print(f"NB: {NB}, NQ: {NQ}, D: {D}")
# points = np.arange(NB*D).reshape(-1, D)
# query_points = np.arange(NQ*D).reshape(-1, D)

points = points.astype(np.float32)
query_points = points.astype(np.float32)

# K = 32 # number of neighbors
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
    # verbose=1,
    # verbose=0,
    n_trees=16,
    # add_bit_random_motion=True,
    add_bit_random_motion=False,
    ensure_valid_indices=False,
    min_tree_children=256,
    max_tree_children=1024) # number of trees
    # min_tree_children=(K+1),
    # max_tree_children=3*(K+1)) # number of trees

# print(indices) # the neighborhood of each point
# print(dist) # the squared distance to each neighbor

# print(f"{time.time() - t_init}")
print(f"{rsfk._last_search_time}")
nne_rate = get_nne_rate(I,indices, max_k=K)
print(f"NNE Rate: {nne_rate}")
negative_indices = np.sum(indices==-1)
if negative_indices > 0:
    raise Exception('{} Negative indices'.format(negative_indices))


print("\n\nIVFFLAT\n")

# '''
# for nprobe in range(5, 101, 5):
for nprobe in range(1, 5):
    print(f"nprobe={nprobe}")
    t_init = time.time()
    res = faiss.StandardGpuResources()  # use a single GPU
    # Set the base and query points as the same 
    xb = np.require(points, np.float32, ['CONTIGUOUS', 'ALIGNED'])
    xq = np.require(query_points, np.float32, ['CONTIGUOUS', 'ALIGNED'])
    nq, d = xq.shape

    # Parameters that control the trade-off between Quality and Time  
    nlist = int(np.sqrt(len(points)))
    # nlist = int(np.sqrt(10000))
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
# '''


print("\n\nIVFPQ\n")
# for nprobe in range(5, 101, 5):
# for nprobe in range(1, 5):
for nprobe in range(1, 21, 2):
    print(f"nprobe={nprobe}")
    t_init = time.time()
    res = faiss.StandardGpuResources()  # use a single GPU
    # Set the base and query points as the same 
    xb = np.require(points, np.float32, ['CONTIGUOUS', 'ALIGNED'])
    xq = np.require(query_points, np.float32, ['CONTIGUOUS', 'ALIGNED'])
    nq, d = xq.shape
    # nlist = int(np.sqrt(10000))
    nlist = int(np.sqrt(len(points)))
    
    m = 2
    quantizer = faiss.IndexFlatL2(d)  # this remains the same
    index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8)

    # Parameters that control the trade-off between Quality and Time  
    # nprobe = 20

    
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