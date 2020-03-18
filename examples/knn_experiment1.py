from gpu_rpfk.RPFK import RPFK

import time
from annoy import AnnoyIndex
import random
from tqdm import tqdm
from multiprocessing import Pool
import multiprocessing
WORKERS = multiprocessing.cpu_count()

import os
import gc

import time
import numpy as np
import pdb

import faiss

# from faiss.datasets import load_sift1M, evaluate

# from vptree import VpTree


from datasets import load_dataset, load_dataset_knn

from utils.knn_compare import get_nne_rate, KnnResult


if __name__ == "__main__":
    knn_result_exp = KnnResult(".","exp1")

    # N = 2048
    # D = 2
    # dataX = np.random.random((N,D)).astype(np.float32)
    # DATA_SET = "MNIST_SKLEARN"
    # DATA_SET = "CIFAR"
    # DATA_SET = "MNIST"
    # DATA_SET = "LUCID_INCEPTION"
    DATA_SET = "AMAZON_REVIEW_ELETRONICS"
    # DATA_SET = "GOOGLE_NEWS300"

    K = 8
    quality_name = "nne"
    dataset_name = DATA_SET
    

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
    
    # '''
    real_sqd_dist, real_indices = load_dataset_knn(DATA_SET, max_k=K)
    real_indices = real_indices[new_indices,:K].astype(np.int)
    real_sqd_dist = real_sqd_dist[new_indices,:K]
    # '''

    if TEST_RPFK:
        for nnef in [0,1,2]:
            if nnef > 0:
                # knn_method_name = "RPFK (NN exploring factor = {})".format(nnef)
                knn_method_name = "RPFK MTC32 (NN exploring factor = {})".format(nnef)
            else:
                # knn_method_name = "RPFK"
                knn_method_name = "RPFK MTC32"

            parameter_name = "n_trees"
            parameter_list = [x+1 for x in range(1,21,2)]
            quality_list = []
            time_list = []
            for n_trees in parameter_list:
                init_t = time.time()
                rpfk = RPFK(K, random_state=0, nn_exploring_factor=nnef,
                            add_bit_random_motion=True)
                indices, dist = rpfk.find_nearest_neighbors(dataX[new_indices],
                                                            max_tree_chlidren=32,
                                                            # max_tree_chlidren=len(dataX),
                                                            max_tree_depth=5000,
                                                            n_trees=n_trees,
                                                            transposed_points=True,
                                                            random_motion_force=0.1,
                                                            # verbose=0)
                                                            # verbose=1)
                                                            verbose=2)
                
                t = time.time() - init_t
                nne_rate = get_nne_rate(real_indices,indices, max_k=K)
                
                time_list.append(t)
                quality_list.append(nne_rate)
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

                print("RPFK NNP: {}".format(nne_rate))
                
            knn_result_exp.add_knn_result(dataset_name, K, knn_method_name, parameter_name, parameter_list,
                                            quality_name, quality_list, time_list)

            # print(np.sum(indices==-1))
            # for i in np.where(indices==-1)[0]:
            #     print(indices[i])



        


    # exit()

    if TEST_ANNOY:
        knn_method_name = "ANNOY"
        parameter_name = "n_trees"
        parameter_list = [x+1 for x in range(1,21,2)]
        quality_list = []
        time_list = []

        for n_trees in parameter_list:
            init_t = time.time()

            f = dataX.shape[1]
            t = AnnoyIndex(f, 'euclidean')  # Length of item vector that will be indexed
            for i,v in enumerate(dataX):
                t.add_item(i, v)

            t.build(n_trees) # 10 trees
            indices = []
            distances = [] 

            gc.collect()

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
            t = time.time() - init_t
            gc.collect()

            for idx,d in ziped_result:
                indices.append(idx)
                distances.append(d)

            gc.collect()

            indices = np.array(indices)
            nne_rate = get_nne_rate(real_indices,indices, max_k=K)
            time_list.append(t)
            quality_list.append(nne_rate)

            # t.save('test.ann')
            print("{} takes {} seconds".format(knn_method_name,t))
            print("{} NNP: {}".format(knn_method_name,nne_rate))
            knn_result_exp.add_knn_result(dataset_name, K, knn_method_name, parameter_name, parameter_list,
                                            quality_name, quality_list, time_list)

            # ...
            gc.collect()

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
        knn_method_name = "IVFFLAT"
        parameter_name = "nprobe"
        parameter_list = [x+1 for x in range(10)]
        quality_list = []
        time_list = []
        
        
        for nprobe in parameter_list:
            quantizer = faiss.IndexFlatL2(d)  # the other index
            index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
            index.nprobe = nprobe              # default nprobe is 1, try a few more
            # here we specify METRIC_L2, by default it performs inner-product search
            assert not index.is_trained
            index.train(xb)
            assert index.is_trained
            init_t = time.time()
            index.add(xb)                  # add may be a bit slower as well
            
            index = faiss.index_cpu_to_gpu(res, 0, index)
            D, I = index.search(xq, K)     # actual search

            t = time.time() - init_t
            nne_rate = get_nne_rate(real_indices,I, max_k=K)

            print("FAISS IVFFLAT (nprobe={}) takes {} seconds".format(nprobe,t))
            print("FAISS IVFFLAT NNE (nprobe={}): {}".format(nprobe,nne_rate))
            quality_list.append(nne_rate)
            time_list.append(t)
        
        knn_result_exp.add_knn_result(dataset_name, K, knn_method_name, parameter_name, parameter_list,
                                      quality_name, quality_list, time_list)
    
        
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
        knn_method_name = "FLATL2"
        parameter_name = "Brute Force"
        parameter_list = [None, None]
        quality_list = []
        time_list = []
        
        init_t = time.time()
        # m=min([x for x in range(1,d) if d%x == 0 and x >= 32])
        index = faiss.IndexFlatL2(d)
        index = faiss.index_cpu_to_gpu(res, 0, index)
                                        # 8 specifies that each sub-vector is encoded as 8 bits
        index.train(xb)
        index.add(xb)
        D, I = index.search(xq, K)     # search
        
        t = time.time() - init_t
        nne_rate = get_nne_rate(real_indices,I, max_k=K)
        
        time_list = [t]*2
        quality_list = [0.0,1.0]
        
        print("FAISS FLATL2 takes {} seconds".format(t))
        print("FAISS FLATL2 NNE: {}".format(nne_rate))
        knn_result_exp.add_knn_result(dataset_name, K, knn_method_name, parameter_name, parameter_list,
                                      quality_name, quality_list, time_list)

    
    knn_result_exp.save()
    knn_result_exp.plot(dataset_name, K, quality_name, dataX,
                        dash_method=["FLATL2"])