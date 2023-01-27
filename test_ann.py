import numpy as np
from gpu_rsfk.RSFK import RSFK
import faiss
import time
from examples.utils.knn_compare import get_nne_rate, create_recall_eps, KnnResult
from examples.datasets import load_dataset, load_dataset_knn, get_dataset_options

from sklearn.model_selection import train_test_split
import pickle


from sklearn.decomposition import PCA

# print(get_dataset_options())
# exit()

np.random.seed(0)

'''
NB = 1000000
NQ = 1000000
# NB = 10000
# NQ = 1000
D = 64

t_init = time.time()
points = np.random.random((NQ,D))
query_points = np.random.random((NQ,D))
'''

# dataset = "MNIST"
# points, points_label, points_query, points_query_label = load_dataset(
#     dataset,
#     return_test_data=True)

run_experiments = True
# '''
# size = 1000000
# size = 2000000
size = 99999999999
dataset = "GOOGLE_NEWS300"
# dataset = "MNIST"

dataX, dataY = load_dataset(dataset)

size = min(size, len(dataX))
dataset = f'{dataset}_{size}'

np.random.shuffle(dataX)

# size = 10000

D = 32
# D = 64
# D = 128
# size = 10000
# dataX = dataX[:size, :D]
dataX = dataX[:size]
dataY = dataY[:size]


X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
pca = PCA(n_components=D)
dataX = pca.fit_transform(dataX)


points, points_query, points_label, points_query_label = train_test_split(
    # dataX, dataY, test_size=0.10, random_state=42)
    dataX, dataY, test_size=0.5, random_state=42)


NB = len(points)
NQ = len(points_query)
D = points.shape[1]
# '''

print(f"NB: {NB}, NQ: {NQ}, D: {D}")
# points = np.arange(NB*D).reshape(-1, D)
# query_points = np.arange(NQ*D).reshape(-1, D)

points = points.astype(np.float32)
query_points = points.astype(np.float32)

K = 32 # number of neighbors
# K = 128 # number of neighbors
# K = 256 # number of neighbors




exp_path = '.'
exp_name = 'knn_experiment_ANN'

knn_result_exp = KnnResult(exp_path, exp_name)



if run_experiments:

    res = faiss.StandardGpuResources()  # use a single GPU
    init_t = time.time()
    index = faiss.IndexFlatL2(D)
    index = faiss.index_cpu_to_gpu(res, 0, index)
    index.train(points)
    index.add(points)
    D, I = index.search(query_points, K)     # search
    t = time.time() - init_t

    print("FAISS FLATL2 takes {} seconds".format(t))

    knn_method_name = "FLATL2"

    knn_result_exp.add_knn_result(
        dataset,
        K, knn_method_name, 'Brute Force', [None, None],
        'accuracy', [0.0, 1.0], [t, t])





    knn_method_name = "ANN-RSFK"



    parameter_list = []
    quality_list = []
    time_list = []

    # for n_trees in range(1,80, 5):
    # for n_trees in [5]:
    for n_trees in [5, 10, 20, 30]:
        rsfk = RSFK(random_state=0)
        indices, dist = rsfk.find_nearest_neighbors_ann(
            points,
            query_points,
            K,
            verbose=2,
            # verbose=1,
            # verbose=0,
            n_trees=n_trees,
            # add_bit_random_motion=True,
            add_bit_random_motion=False,
            ensure_valid_indices=False,
            min_tree_children=32,
            max_tree_children=256) # number of trees
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

        parameter_list.append(n_trees)
        quality_list.append(nne_rate)
        time_list.append(rsfk._last_search_time)



    knn_result_exp.add_knn_result(
        dataset,
        K, knn_method_name, 'n_trees', parameter_list,
        'accuracy', quality_list, time_list)




    print("\n\nIVFFLAT\n")
    knn_method_name = 'IVFFLAT'

    parameter_list = []
    quality_list = []
    time_list = []

    # '''
    # for nprobe in range(5, 101, 5):
    for nprobe in range(1, 21):
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
        ti = time.time() - t_init
        print(f"Index: {ti}")
        t_init = time.time()

        dist, indices = index.search(xq, K)     # actual search

        # print(indices)
        # print(dist)
        tq = time.time() - t_init
        print(f"Query: {tq}")
        nne_rate = get_nne_rate(I,indices, max_k=K)
        print(f"NNE Rate: {nne_rate}")

        parameter_list.append(nprobe)
        quality_list.append(nne_rate)
        time_list.append(ti+tq)


    # '''

    knn_result_exp.add_knn_result(
        dataset,
        K, knn_method_name, 'nprobe', parameter_list,
        'accuracy', quality_list, time_list)



    print("\n\nIVFPQ\n")
    knn_method_name = 'IVFPQ'

    parameter_list = []
    quality_list = []
    time_list = []

    # for nprobe in range(5, 101, 5):
    # for nprobe in range(1, 5):
    for nprobe in range(1, 21, 2):
    # for nprobe in range(1, 21):
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
        ti = time.time() - t_init
        print(f"Index: {ti}")
        t_init = time.time()

        dist, indices = index.search(xq, K)     # actual search

        # print(indices)
        # print(dist)
        tq = time.time() - t_init
        print(f"Query: {tq}")
        nne_rate = get_nne_rate(I,indices, max_k=K)
        print(f"NNE Rate: {nne_rate}")

        parameter_list.append(nprobe)
        quality_list.append(nne_rate)
        time_list.append(ti+tq)


    knn_result_exp.add_knn_result(
        dataset,
        K, knn_method_name, 'nprobe', parameter_list,
        'accuracy', quality_list, time_list)

    exit()

    knn_result_exp.save()

knn_result_exp.plot(dataset, K, 'accuracy', points_query,
                            dash_method=["FLATL2"])