from gpu_rsfk.RSFK import RSFK

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

import argparse

# from faiss.datasets import load_sift1M, evaluate

# from vptree import VpTree


from datasets import load_dataset, load_dataset_knn, get_dataset_options

from utils.knn_compare import get_nne_rate, create_recall_eps, KnnResult


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parameters used to init experiments")
    parser.add_argument("-n", "--name", help="Experiment name used to create json log",
                        type=str, default="exp4")
    parser.add_argument("-p", "--path", help="Experiment path used to create json log",
                        type=str, default=".")
    parser.add_argument("-v", "--rsfk_verbose", help="RSFK verbose level ",
                        type=int, default=1)
    parser.add_argument("-k", "--n_neighbors", help="Number of neighbors (K) used in KNN",
                        type=int, default=32)
    parser.add_argument("--mntc", help="Max Tree Chlidren parameter in RSFK",
                        type=int, default=-1)
    parser.add_argument("--mxtc", help="Max Tree Chlidren parameter in RSFK",
                        type=int, default=-1)
    parser.add_argument("-d", "--dataset", help="Dataset name",
                        type=str, default="MNIST_SKLEARN",
                        choices=get_dataset_options())
    parser.add_argument("--recall_eps_val", help="Recall-e value",
                        type=float, default=0.01)
    
    parser.add_argument('-e', '--nnexp_factor_list', type=int, nargs='+',
                        default=[0],
                        help='Nearest Neighbors Exploring factor list to test')

    parser.add_argument('-t','--n_trees_list', type=int, nargs='+',
                        default=[x for x in range(1,29,2)],
                        help='List of total of trees to test')
    # parser.add_argument("-nn", "--numneigh", help="Number of nearest neighbors used in KNN",type=int,default=DEFAULT_NUM_NEIGHBORS)
    # parser.add_argument("-p", "--perplexity", help="Perplexity",type=float,default=DEFAULT_PERPLEXITY)
    # parser.add_argument("-ms", "--maxsamples", help="Max samples from dataset. Default is use all samples",
    #                     type=int,default=DEFAULT_MAX_SAMPLES)

    # parser.add_argument('--pre_knn', dest='pre_knn', action='store_true', default=False,
    #                     help="Fetch pre KNN already computed and stored in file")
    # parser.add_argument('--cpu_knn', dest='cpu_knn', action='store_true', default=False,
    #                     help="Pre compute KNN in CPU. No effect if pre_knn is enabled")

    # parser.add_argument('--save', dest='save_log', action='store_true', default=False,
    #                     help="Save log in results")
    # parser.add_argument('--plot', dest='plot', action='store_true', default=False,
    #                     help="Plot embedding into 2 dimensions")
    
    parser.add_argument('--save_plot', dest='save_plot', action='store_true', default=False,
                        help="Create and save a plot into a png comparing different executions from log file")

    parser.add_argument('--skip_save', dest='skip_save', action='store_true', default=False,
                        help="Doesnt change log file")


    parser.add_argument('--test_rsfk', dest='test_rsfk', action='store_true', default=False,
                        help="Test Random Projection Forest KNN")

    # parser.add_argument('--test_rsfk', dest='test_rsfk', action='store_true', default=False,
                        # help="Test Random Projection Forest KNN")

    parser.add_argument('--test_annoy', dest='test_annoy', action='store_true', default=False,
                        help="Test ANNOY")

    parser.add_argument('--test_ivfflat', dest='test_ivfflat', action='store_true', default=False,
                        help="Test FAISS-IVFFLAT")
    parser.add_argument('--test_flatl2', dest='test_flatl2', action='store_true', default=False,
                        help="Test FAISS-FLATL2")


    parser.add_argument('--sanity_check', dest='sanity_check', action='store_true', default=False,
                        help="Test FAISS-FLATL2")


    args = parser.parse_args()

    exp_name = args.name
    exp_path = args.path
    rsfk_verbose = args.rsfk_verbose
    sanity_check = args.sanity_check

    # embsize = args.embsize
    DATA_SET = args.dataset
    # maxsamples = args.maxsamples
    # numneigh = args.numneigh
    # perplexity = args.perplexity
    
    # pre_knn = args.pre_knn
    # cpu_knn = args.cpu_knn
    # save_log = args.save_log
    # plot = args.plot

    save_plot = args.save_plot
    skip_save = args.skip_save
    K = args.n_neighbors
    min_tree_children = args.mntc
    max_tree_children = args.mxtc
    nnexp_factor_list = args.nnexp_factor_list
    n_trees_list = args.n_trees_list
    knn_result_exp = KnnResult(exp_path, exp_name)
    # max_tree_children = 126
    # max_tree_children = 30

    # N = 2048
    # D = 2
    # dataX = np.random.random((N,D)).astype(np.float32)
    # DATA_SET = "MNIST_SKLEARN"
    # DATA_SET = "CIFAR"
    # DATA_SET = "MNIST"
    # DATA_SET = "LUCID_INCEPTION"
    # DATA_SET = "AMAZON_REVIEW_ELETRONICS"
    # DATA_SET = "GOOGLE_NEWS300"

    quality_name = "nne"
    # quality_name = "recall_eps"
    recall_eps_val = args.recall_eps_val

    if quality_name == "nne":
        quality_function = get_nne_rate
    if quality_name == "recall_eps":
        quality_function = create_recall_eps(recall_eps_val)
        quality_name = "{}_{}".format(quality_name, recall_eps_val)

    dataset_name = DATA_SET
    

    TEST_RSFK = args.test_rsfk
    TEST_ANNOY = args.test_annoy
    TEST_IVFFLAT = False
    TEST_IVFFLAT10 = args.test_ivfflat
    TEST_IVFPQ = False
    TEST_IVFPQ10 = False
    TEST_HNSWFLAT = False
    TEST_FLATL2 = args.test_flatl2

    # TEST_RSFK = True
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

    if DATA_SET != "GOOGLE_NEWS300" or K <=32:
        t = time.time()
        real_sqd_dist, real_indices = load_dataset_knn(DATA_SET, max_k=K)
    else:
        def dist_mean(real_indices, indices, real_dist, dist,
                     random_state=0, max_k=32, verbose=0):
            return np.average(dist, weights=np.ones_like(dist)/len(dist))
        quality_function = dist_mean
        quality_name = "dist_mean"
        
        real_sqd_dist = np.zeros((len(dataX),K))
        real_indices = np.zeros((len(dataX),K))
    
    real_indices = real_indices[new_indices,:K].astype(np.int)
    real_sqd_dist = real_sqd_dist[new_indices,:K]
    # '''


    # knn_method_name = "RSFK-Atomic"
    knn_method_name = "RSFK-Tiles (1024 threads per block)"
    # knn_method_name = "RSFK-Tiles (512 threads per block)"
    # knn_method_name = "RSFK-Diagonal"
    # knn_method_name = "FAISS"
    # knn_method_name = "ANNOY"
    
    if TEST_RSFK:
        # for nnef in [0]:
        # for nnef in [1]:
        # for nnef in [1,2]:
        # for nnef in [0,1,2]:
        for nnef in nnexp_factor_list:
        # for nnef in [3]:
            # if nnef > 0:
                # if max_tree_children == -1:
                #     knn_method_name = "RSFK (NN exploring factor = {})".format(nnef)
                # else:
                #     knn_method_name = "RSFK MNTC{} MXTC{} (NN exploring factor = {})".format(min_tree_children, max_tree_children, nnef)
            # else:
            #     if max_tree_children == -1:
            #         knn_method_name = "RSFK"
            #     else:
            #         knn_method_name = "RSFK MNTC{} MXTC{}".format(min_tree_children, max_tree_children)

            if nnef > 0:
                knn_method_name = "RSFK (NN Exploring factor = {})".format(nnef)


            print("Testing knn method: {}".format(knn_method_name))
            parameter_name = "n_trees"
            parameter_list = n_trees_list
            # parameter_list = [10]
            quality_list = []
            time_list = []
            for n_trees in parameter_list:
                init_t = time.time()
                rsfk = RSFK(random_state=0)
                indices, dist = rsfk.find_nearest_neighbors(dataX[new_indices],
                                                            K,
                                                            min_tree_children=min_tree_children,
                                                            max_tree_children=max_tree_children,
                                                            # max_tree_children=len(dataX),
                                                            max_tree_depth=5000,
                                                            n_trees=n_trees,
                                                            random_motion_force=0.1,
                                                            ensure_valid_indices=True,
                                                            nn_exploring_factor=nnef,
                                                            add_bit_random_motion=False,
                                                            # verbose=0)
                                                            # verbose=1)
                                                            # verbose=2)
                                                            verbose=rsfk_verbose)
                # t = time.time() - init_t
                t = rsfk._last_search_time # Ignore data initialization time

                nne_rate = quality_function(real_indices,indices, real_sqd_dist, dist, max_k=K)
                
                time_list.append(t)
                quality_list.append(nne_rate)
                print("RSFK Time: {}".format(t), flush=True)
                print("RSFK NNP: {}".format(nne_rate), flush=True)
                print("")
                # exit()
                
                # idx = np.arange(len(dataX)).reshape((-1,1))
                # print(np.append(idx,indices,axis=1), np.sort(dist,axis=1))
                # print(np.append(idx,real_indices,axis=1), np.sort(real_sqd_dist,axis=1))

                # print(indices.shape, dist.shape)
                # print(real_indices.shape, real_sqd_dist.shape)

                # Sanity check
                if sanity_check:
                    negative_indices = np.sum(indices==-1)
                    if negative_indices > 0:
                        raise Exception('{} Negative indices'.format(negative_indices))
                
                
                
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

            t.build(n_trees, n_jobs=-1) # 10 trees
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
            distances = np.array(distances)
            nne_rate = quality_function(real_indices,indices, real_sqd_dist, distances, max_k=K)
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
        print("FAISS IVFFLAT NNE: {}".format(quality_function(real_indices,I, max_k=K)))

    if TEST_IVFFLAT10:
        knn_method_name = "IVFFLAT"
        parameter_name = "nprobe"
        parameter_list = [x+1 for x in range(20)]
        # parameter_list = [20]
        quality_list = []
        time_list = []
        
        
        for nprobe in parameter_list:
            init_t = time.time()
            quantizer = faiss.IndexFlatL2(d)  # the other index
            index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
            index.nprobe = nprobe              # default nprobe is 1, try a few more
            # here we specify METRIC_L2, by default it performs inner-product search
            assert not index.is_trained
            index.train(xb)
            assert index.is_trained

            index.add(xb)                  # add may be a bit slower as well
            
            index = faiss.index_cpu_to_gpu(res, 0, index)
            D, I = index.search(xq, K)     # actual search

            t = time.time() - init_t
            nne_rate = quality_function(real_indices, I, real_sqd_dist, D, max_k=K)

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
        print("FAISS IVFPQ NNE: {}".format(quality_function(real_indices,I, max_k=K)))
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
        print("FAISS IVFPQ NNE (nprob=10): {}".format(quality_function(real_indices,I, max_k=K)))
    if TEST_HNSWFLAT:
        init_t = time.time()
        m=16
        index = faiss.IndexHNSWFlat(d,m)
        # index = faiss.index_cpu_to_gpu(res, 0, index)
        index.train(xb)
        index.add(xb)
        D, I = index.search(xq, K)     # search
        print("FAISS HNSWFLAT takes {} seconds".format(time.time() - init_t))
        print("FAISS HNSWFLAT NNE: {}".format(quality_function(real_indices,I, max_k=K)))
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
        nne_rate = quality_function(real_indices, I, real_sqd_dist, D, max_k=K)
        
        
        time_list = [t]*2
        quality_list = [0.0,1.0]
        
        print("FAISS FLATL2 takes {} seconds".format(t))
        print("FAISS FLATL2 NNE: {}".format(nne_rate))
        knn_result_exp.add_knn_result(dataset_name, K, knn_method_name, parameter_name, parameter_list,
                                      quality_name, quality_list, time_list)

    
    if not skip_save:
        knn_result_exp.save()
    
    if save_plot:
        # knn_result_exp.plot([dataset_name], K, quality_name, dataX,
        #                     dash_method=["FLATL2"], baseline="IVFFLAT",
        #                     method_list=["IVFFLAT", "RSFK-Tiles"])

        knn_result_exp.plot(["ATSNE_IMAGENET", "GOOGLE_NEWS300"], K, quality_name, dataX,
                            dash_method=["FLATL2"], baseline="IVFFLAT",
                            method_list=["IVFFLAT", "RSFK-Tiles (NN Exploring factor = 1)", "RSFK-Tiles"])