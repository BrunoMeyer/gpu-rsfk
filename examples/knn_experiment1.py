from gpu_rpfk.RPFK import RPFK

import time
from annoy import AnnoyIndex
import random
from tqdm import tqdm
from multiprocessing import Pool
import multiprocessing
WORKERS = multiprocessing.cpu_count()

import os
import time
import numpy as np
import pdb

import faiss

# from faiss.datasets import load_sift1M, evaluate

# from vptree import VpTree


from datasets import load_dataset, load_dataset_knn


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
    # return qnx
    return rnx


if __name__ == "__main__":
    
    K = 32
    
    TEST_RPFK = False
    TEST_ANNOY = False
    TEST_IVFFLAT = False
    TEST_IVFFLAT10 = False
    TEST_IVFPQ = False
    TEST_IVFPQ10 = False
    TEST_HNSWFLAT = False
    TEST_FLATL2 = False

    TEST_RPFK = True
    # TEST_ANNOY = True
    # TEST_IVFFLAT = True
    # TEST_IVFFLAT10 = True
    # TEST_IVFPQ = True
    # TEST_IVFPQ10 = True
    # TEST_HNSWFLAT = True
    # TEST_FLATL2 = True

    # N = 2048
    # D = 2
    # dataX = np.random.random((N,D)).astype(np.float32)
    DATA_SET = "MNIST_SKLEARN"
    # DATA_SET = "CIFAR"
    # DATA_SET = "MNIST"
    # DATA_SET = "LUCID_INCEPTION"
    # DATA_SET = "AMAZON_REVIEW_ELETRONICS"

    dataX, dataY = load_dataset(DATA_SET)
    print("dataX.shape: {}".format(dataX.shape))
    print("K = {}".format(K))
    new_indices = np.arange(len(dataX))
    # np.random.shuffle(new_indices)

    '''
    neigh = NearestNeighbors(K, n_jobs=-1)
    neigh.fit(dataX)
    real_sqd_dist, real_indices = neigh.kneighbors(dataX)
    '''
    
    #'''
    real_sqd_dist, real_indices = load_dataset_knn(DATA_SET, max_k=K)
    real_indices = real_indices[new_indices,:K].astype(np.int)
    real_sqd_dist = real_sqd_dist[new_indices,:K]
    #'''

    if TEST_RPFK:
        rpfk = RPFK(K, random_state=0, nn_exploring_factor=1,
                    add_bit_random_motion=True)
        indices, dist = rpfk.find_nearest_neighbors(dataX[new_indices],
                                                    max_tree_chlidren=-1,
                                                    # max_tree_chlidren=len(dataX),
                                                    max_tree_depth=5000,
                                                    n_trees=10,
                                                    transposed_points=True,
                                                    random_motion_force=0.01,
                                                    # verbose=0)
                                                    # verbose=1)
                                                    verbose=2)

        # exit()
        
        # idx = np.arange(len(dataX)).reshape((-1,1))
        # print(np.append(idx,indices,axis=1), np.sort(dist,axis=1))
        # print(np.append(idx,real_indices,axis=1), np.sort(real_sqd_dist,axis=1))

        # print(indices.shape, dist.shape)
        # print(real_indices.shape, real_sqd_dist.shape)

        # Sanity check
        negative_indices = np.sum(indices==-1)
        if negative_indices > 0:
            raise Exception('{} Negative indices'.format(negative_indices))

        print("RPFK NNP: {}".format(get_nne_rate(real_indices,indices, max_k=K)))


        # print(np.sum(indices==-1))
        # for i in np.where(indices==-1)[0]:
        #     print(indices[i])



        


    # exit()

    if TEST_ANNOY:
        init_t = time.time()

        f = dataX.shape[1]
        t = AnnoyIndex(f, 'euclidean')  # Length of item vector that will be indexed
        for i,v in enumerate(dataX):
            t.add_item(i, v)

        t.build(50) # 10 trees
        indices = []
        distances = [] 

        def individual_query(i):
            return t.get_nns_by_item(i, K,
                                    #  search_k=1,
                                     search_k=-1,
                                     include_distances=True)
        p = Pool(WORKERS)
        arguments = [i for i in range(len(dataX))]
        ziped_result = list(tqdm(p.imap(individual_query, arguments),
                       total=len(arguments),
                       desc="Computing ANNOY"))
        p.terminate()


        for idx,d in ziped_result:
            indices.append(idx)
            distances.append(d)
        
        indices = np.array(indices)

        # t.save('test.ann')
        print("ANNOY takes {} seconds".format(time.time() - init_t))
        print("ANNOY NNP: {}".format(get_nne_rate(real_indices,indices, max_k=K)))

        # ...

        # u = AnnoyIndex(f, 'angular')
        # u.load('test.ann') # super fast, will just mmap the file
        # print(u.get_nns_by_item(0, 1000)) # will find the 1000 nearest neighbors




    # print("PRESS ENTER TO CONTINUE"); input()

    # Copyright (c) Facebook, Inc. and its affiliates.
    #
    # This source code is licensed under the MIT license found in the
    # LICENSE file in the root directory of this source tree.

    #!/usr/bin/env python2

    if TEST_IVFFLAT or TEST_IVFFLAT10 or TEST_IVFPQ or TEST_IVFPQ10 or TEST_HNSWFLAT or TEST_FLATL2:

        res = faiss.StandardGpuResources()  # use a single GPU
        # xb, xq, xt, gt = load_sift1M()
        xb = np.require(dataX, np.float32, ['CONTIGUOUS', 'ALIGNED'])
        xq = np.require(dataX, np.float32, ['CONTIGUOUS', 'ALIGNED'])

        nq, d = xq.shape

        nlist = int(np.sqrt(nq))

    if TEST_IVFFLAT:
        init_t = time.time()
        quantizer = faiss.IndexFlatL2(d)  # the other index
        index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)

        # here we specify METRIC_L2, by default it performs inner-product search
        assert not index.is_trained
        index.train(xb)
        assert index.is_trained
        index.add(xb)                  # add may be a bit slower as well
        index.nprobe = 1              # default nprobe is 1, try a few more
        index = faiss.index_cpu_to_gpu(res, 0, index)
        D, I = index.search(xq, K)     # actual search
        print("FAISS IVFFLAT takes {} seconds".format(time.time() - init_t))
        print("FAISS IVFFLAT NNE: {}".format(get_nne_rate(real_indices,I, max_k=K)))
    if TEST_IVFFLAT10:
        quantizer = faiss.IndexFlatL2(d)  # the other index
        index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
        index.nprobe = 20              # default nprobe is 1, try a few more
        # here we specify METRIC_L2, by default it performs inner-product search
        assert not index.is_trained
        index.train(xb)
        assert index.is_trained
        init_t = time.time()
        index.add(xb)                  # add may be a bit slower as well
        
        index = faiss.index_cpu_to_gpu(res, 0, index)
        D, I = index.search(xq, K)     # actual search
        print("FAISS IVFFLAT (nprob=10) takes {} seconds".format(time.time() - init_t))
        print("FAISS IVFFLAT NNE (nprob=10): {}".format(get_nne_rate(real_indices,I, max_k=K)))
    if TEST_IVFPQ:
        init_t = time.time()
        # m=min([x for x in range(1,d) if d%x == 0 and x >= 32])
        m = 5
        quantizer = faiss.IndexFlatL2(d)  # this remains the same
        index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8)
                                        # 8 specifies that each sub-vector is encoded as 8 bits
        index.nprobe = 1              # make comparable with experiment above
        index.train(xb)
        index.add(xb)
        index = faiss.index_cpu_to_gpu(res, 0, index)
        D, I = index.search(xq, K)     # search
        print("FAISS IVFPQ takes {} seconds".format(time.time() - init_t))
        print("FAISS IVFPQ NNE: {}".format(get_nne_rate(real_indices,I, max_k=K)))
    if TEST_IVFPQ10:
        init_t = time.time()
        m = 4
        # m=min([x for x in range(1,d) if d%x == 0 and x >= 32])
        quantizer = faiss.IndexFlatL2(d)  # this remains the same
        index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8)
        index.nprobe = 1              # make comparable with experiment above
                                        # 8 specifies that each sub-vector is encoded as 8 bits
        index.train(xb)
        index.add(xb)
        index.nprobe = 10              # make comparable with experiment above
        index = faiss.index_cpu_to_gpu(res, 0, index)
        D, I = index.search(xq, K)     # search
        print("FAISS IVFPQ (nprob=10) takes {} seconds".format(time.time() - init_t))
        print("FAISS IVFPQ NNE (nprob=10): {}".format(get_nne_rate(real_indices,I, max_k=K)))
    if TEST_HNSWFLAT:
        init_t = time.time()
        m=16
        index = faiss.IndexHNSWFlat(d,m)
        # index = faiss.index_cpu_to_gpu(res, 0, index)
        index.train(xb)
        index.add(xb)
        D, I = index.search(xq, K)     # search
        print("FAISS HNSWFLAT takes {} seconds".format(time.time() - init_t))
        print("FAISS HNSWFLAT NNE: {}".format(get_nne_rate(real_indices,I, max_k=K)))
    if TEST_FLATL2:
        init_t = time.time()
        # m=min([x for x in range(1,d) if d%x == 0 and x >= 32])
        index = faiss.IndexFlatL2(d)
        index = faiss.index_cpu_to_gpu(res, 0, index)
                                        # 8 specifies that each sub-vector is encoded as 8 bits
        index.train(xb)
        index.add(xb)
        D, I = index.search(xq, K)     # search
        print("FAISS FLATL2 takes {} seconds".format(time.time() - init_t))
        print("FAISS FLATL2 NNE: {}".format(get_nne_rate(real_indices,I, max_k=K)))


    '''
    print("benchmark")

    for lnprobe in range(10):
        nprobe = 1 << lnprobe
        index.setNumProbes(nprobe)
        t, r = evaluate(index, xq, gt, 100)

        print("nprobe=%4d %.3f ms recalls= %.4f %.4f %.4f" % (nprobe, t, r[1], r[10], r[100]))
    '''

    '''


    init_t = time.time()
    tree = VpTree(dataX)
    
    vpt_distances, vpt_indices = tree.getNearestNeighborsBatch(dataX, K) # split the work between threads
    vpt_time = time.time() - init_t
    
    
    print(indices[0])
    print(vpt_indices[0])
    print(real_indices[0])
    print("")
    print(indices[500])
    print(vpt_indices[500])
    print(real_indices[500])
    print("")
    print(np.sort(dist[0]))
    print(np.array(vpt_distances[0],dtype=np.float32)**2)
    print(real_sqd_dist[0])
    print("")
    print(np.sort(dist[500]))
    print(np.array(vpt_distances[500],dtype=np.float32)**2)
    print(real_sqd_dist[500])

    print("VPT time: {}".format(vpt_time))
    print("VPT NNP: {}".format(get_nne_rate(real_indices,vpt_indices, max_k=K)))
    '''
