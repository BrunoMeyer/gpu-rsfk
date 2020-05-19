from gpu_rpfk.RPFK import RPFK
from tsnecuda import TSNE

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
import json

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
Axes3D = Axes3D  # pycharm auto import
# from matplotlib import rc
# rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
# rc('text', usetex=True)


# from faiss.datasets import load_sift1M, evaluate

# from vptree import VpTree


from datasets import load_dataset, load_dataset_knn, get_dataset_options

from utils.knn_compare import get_nne_rate, create_recall_eps, KnnResult
from test_sklearn import get_nne_rate as get_nne_rate_tsne


def plot_emb(emb, fig_name="emb", fig_title="", save_fig=True, plot_fig=False):
    fig = plt.figure(figsize=(8*2,6*2))
    if emb.shape[1] == 3:
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = fig.add_subplot(111)
    
    set_dataY = set(dataY)
    if len(set_dataY) > 1:
        for y in set(dataY):
            aux = emb[dataY==y]
            if emb.shape[1] == 3:
                ax.scatter(aux[:,0],aux[:,1],aux[:,2], alpha=0.4, s=0.1, label=y)
            else:
                ax.scatter(aux[:,0],aux[:,1], alpha=0.8, s=1, label=y)
    else:
        if emb.shape[1] == 3:
            ax.scatter(emb[:,0], emb[:,1], emb[:,2], alpha=0.4, s=0.1)
        else:
            ax.scatter(emb[:,0], emb[:,1], alpha=0.4, s=0.1)

    ax.set_title(fig_title)
    
    ax = plt.gca()
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    if emb.shape[1] == 3:
        ax.zaxis.set_ticklabels([])

    for line in ax.xaxis.get_ticklines():
        line.set_visible(False)
    for line in ax.yaxis.get_ticklines():
        line.set_visible(False)
    if emb.shape[1] == 3:
        for line in ax.zaxis.get_ticklines():
            line.set_visible(False)

    if plot_fig:
        plt.show(fig)
    if save_fig:
        print("{}.png".format(fig_name))
        plt.savefig("{}.png".format(fig_name), bbox_inches='tight')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parameters used to init experiments")
    parser.add_argument("-n", "--name", help="Experiment name used to create json log",
                        type=str, default="exp2")
    parser.add_argument("-p", "--path", help="Experiment path used to create json log",
                        type=str, default=".")
    parser.add_argument("-v", "--rpfk_verbose", help="RPFK verbose level ",
                        type=int, default=1)
    parser.add_argument("-k", "--n_neighbors", help="Number of neighbors (K) used in KNN",
                        type=int, default=32)
    parser.add_argument("--mntc", help="Max Tree Chlidren parameter in RPFK",
                        type=int, default=-1)
    parser.add_argument("--mxtc", help="Max Tree Chlidren parameter in RPFK",
                        type=int, default=-1)
    parser.add_argument("-d", "--dataset", help="Dataset name",
                        type=str, default="MNIST_SKLEARN",
                        choices=get_dataset_options())
    parser.add_argument("--recall_eps_val", help="Recall-e value",
                        type=float, default=0.01)
    
    parser.add_argument('-e', '--nnexp_factor_list', type=int, nargs='+',
                        default=[0,1,2],
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


    parser.add_argument('--test_rpfk', dest='test_rpfk', action='store_true', default=False,
                        help="Test Random Projection Forest KNN")
    parser.add_argument('--test_ivfflat', dest='test_ivfflat', action='store_true', default=False,
                        help="Test FAISS-IVFFLAT")
    parser.add_argument('--test_flatl2', dest='test_flatl2', action='store_true', default=False,
                        help="Test FAISS-FLATL2")


    parser.add_argument('--sanity_check', dest='sanity_check', action='store_true', default=False,
                        help="Test FAISS-FLATL2")


    args = parser.parse_args()

    exp_name = args.name
    exp_path = args.path
    rpfk_verbose = args.rpfk_verbose
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

    # knn_result_exp = KnnResult(exp_path, exp_name)


    quality_name = "nne"
    # quality_name = "recall_eps"
    recall_eps_val = args.recall_eps_val

    if quality_name == "nne":
        quality_function = get_nne_rate
    if quality_name == "recall_eps":
        quality_function = create_recall_eps(recall_eps_val)
        quality_name = "{}_{}".format(quality_name, recall_eps_val)

    dataset_name = DATA_SET
    

    TEST_RPFK = args.test_rpfk
    TEST_ANNOY = False
    TEST_IVFFLAT = False
    TEST_IVFFLAT10 = args.test_ivfflat
    TEST_IVFPQ = False
    TEST_IVFPQ10 = False
    TEST_HNSWFLAT = False
    TEST_FLATL2 = args.test_flatl2

    # TEST_RPFK = True
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

    if DATA_SET != "GOOGLE_NEWS300":
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

    # PERPLEXITY = K/3
    PERPLEXITY = 50
    EMB_SIZE = 3
    MAX_ITE = 1000
    
    if DATA_SET == "AMAZON_REVIEW_ELETRONICS":
        LEARNING_RATE = 3000
        EARLY_EXAGGERATION = 24.0
        if EMB_SIZE == 2:
            LEARNING_RATE = 2000
            EARLY_EXAGGERATION = 2.0

    if DATA_SET == "LUCID_INCEPTION":
        LEARNING_RATE = 100
        EARLY_EXAGGERATION = 50.0
    if DATA_SET == "MNIST_SKLEARN":
        LEARNING_RATE = 10
        EARLY_EXAGGERATION = 24.0
    if DATA_SET == "GOOGLE_NEWS300":
        LEARNING_RATE = 3000
        EARLY_EXAGGERATION = 24.0
    if DATA_SET == "MNIST":
        LEARNING_RATE = 100
        EARLY_EXAGGERATION = 24.0
    if DATA_SET == "CIFAR":
        LEARNING_RATE = 100
        EARLY_EXAGGERATION = 24.0
    
    MAGNITUDE_FACTOR = 5.0
    EARLY_EXAGGERATION = 2.0
    

    tsne_arg = {
                "early_exaggeration":EARLY_EXAGGERATION,
                "magnitude_factor":MAGNITUDE_FACTOR,
                "warpwidth": 4,
                "random_seed":2,
                "verbose":2,
                "n_iter":MAX_ITE,
                "theta":0.5,
                "num_neighbors": K,
                "learning_rate":LEARNING_RATE,
                "print_interval":100,
                "n_components":EMB_SIZE,
                "perplexity": PERPLEXITY}

    tsne = TSNE(**tsne_arg)
    fig, ax = plt.subplots(figsize=(16, 8))
    
    log_obj = {}
    if TEST_RPFK:
        # for nnef in [0]:
        # for nnef in [1]:
        # for nnef in [1,2]:
        # for nnef in [0,1,2]:
        for nnef in nnexp_factor_list:
        # for nnef in [3]:
            if nnef > 0:
                if max_tree_children == -1:
                    knn_method_name = "RPFK (NN exploring factor = {})".format(nnef)
                else:
                    knn_method_name = "RPFK MNTC{} MXTC{} (NN exploring factor = {})".format(min_tree_children, max_tree_children, nnef)
            else:
                if max_tree_children == -1:
                    knn_method_name = "RPFK"
                else:
                    knn_method_name = "RPFK MNTC{} MXTC{}".format(min_tree_children, max_tree_children)

            print("Testing knn method: {}".format(knn_method_name))
            parameter_name = "n_trees"
            parameter_list = n_trees_list
            # parameter_list = [10]
            quality_list = []
            time_list = []


            for n_trees in parameter_list:
                init_t = time.time()
                rpfk = RPFK(random_state=0)
                indices, dist = rpfk.find_nearest_neighbors(dataX[new_indices],
                                                            K,
                                                            min_tree_children=min_tree_children,
                                                            max_tree_children=max_tree_children,
                                                            # max_tree_children=len(dataX),
                                                            max_tree_depth=5000,
                                                            n_trees=n_trees,
                                                            random_motion_force=0.1,
                                                            ensure_valid_indices=True,
                                                            nn_exploring_factor=nnef,
                                                            add_bit_random_motion=True,
                                                            # verbose=0)
                                                            # verbose=1)
                                                            # verbose=2)
                                                            verbose=rpfk_verbose)
                # t = time.time() - init_t
                t = rpfk._last_search_time # Ignore data initialization time

                nne_rate = quality_function(real_indices,indices, real_sqd_dist, dist, max_k=K)
                
                time_list.append(t)
                print("RPFK Time: {}".format(t), flush=True)
                print("RPFK NNP: {}".format(nne_rate), flush=True)

                p = tsne.fit_transform(dataX, pre_knn=(indices, dist))
                tsne_nne = get_nne_rate_tsne(dataX, p, max_k=K, pre_knn=(real_sqd_dist, real_indices))
                quality_list.append([nne_rate,tsne_nne])
                print("TSNE NNE: {}".format(tsne_nne))

                # plot_emb(p,
                #          fig_name="knn_experiment2/{}_K{}_{}trees".format(dataset_name, K, n_trees),
                #          fig_title="t-SNE result with KNN error = {}\n".format(tsne_nne)+
                #                    r"$R_{\mathrm{NX}}(K)$ = "+str(nne_rate))

                plot_emb(p,fig_name="knn_experiment2/{}_K{}_{}trees".format(dataset_name, K, n_trees))
                
                # fig, ax = plt.subplots(figsize=(16, 8))
                # for y in set(dataY):
                #     ax.scatter(p[dataY==y,0], p[dataY==y,1], label=y, s=0.3)
                # plt.show()

                # exit()

                # Sanity check
                # if sanity_check:
                #     negative_indices = np.sum(indices==-1)
                #     if negative_indices > 0:
                #         raise Exception('{} Negative indices'.format(negative_indices))
                
                print("")

            log_obj["RPSK"] = {
                "parameter_list": parameter_list,
                "quality_list": quality_list,
                "time_list": time_list
            }    
            
                
            
            quality_list = np.array(quality_list)
            ax.plot(quality_list[:,0], quality_list[:,1], "-|", label="RSFK")
            # ax.set_xlabel('KNN Nearest Neighbor Preservation')
            # ax.set_ylabel('T-SNE Nearest Neighbor Preservation')
            # ax.set_title('K = {}'.format(K))
            # fig.savefig("knn_experiment2/{}_K{}.pdf".format(dataset_name, K))



        



    if TEST_IVFFLAT or TEST_IVFFLAT10 or TEST_IVFPQ or TEST_IVFPQ10 or TEST_HNSWFLAT or TEST_FLATL2:

        res = faiss.StandardGpuResources()  # use a single GPU
        # xb, xq, xt, gt = load_sift1M()
        xb = np.require(dataX, np.float32, ['CONTIGUOUS', 'ALIGNED'])
        xq = np.require(dataX, np.float32, ['CONTIGUOUS', 'ALIGNED'])

        nq, d = xq.shape

        nlist = int(np.sqrt(nq))

    if TEST_IVFFLAT10:
        knn_method_name = "IVFFLAT"
        parameter_name = "nprobe"
        parameter_list = [x+1 for x in range(10)]
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

            print("FAISS IVFFLAT (nprobe={}) takes {} seconds".format(nprobe,t), flush=True)
            print("FAISS IVFFLAT NNE (nprobe={}): {}".format(nprobe,nne_rate), flush=True)
            time_list.append(t)

            p = tsne.fit_transform(dataX, pre_knn=(I, D))
            plot_emb(p,fig_name="knn_experiment2/{}_K{}_{}nprobe".format(dataset_name, K, nprobe))
            tsne_nne = get_nne_rate_tsne(dataX, p, max_k=K, pre_knn=(real_sqd_dist, real_indices))
            quality_list.append([nne_rate,tsne_nne])
        
        log_obj["IVFFLAT"] = {
            "parameter_list": parameter_list,
            "quality_list": quality_list,
            "time_list": time_list
        }
        
        quality_list = np.array(quality_list)
        ax.plot(quality_list[:,0], quality_list[:,1], "-|", label="IVFFLAT")


    # with open("knn_experiment2/{}_K{}.json".format(dataset_name, K), "w+") as outfile:
    #     json.dump(log_obj, outfile)
    
    # ax.set_xlabel('KNN Nearest Neighbor Preservation')
    # ax.set_ylabel('T-SNE Nearest Neighbor Preservation')
    # ax.set_title('K = {}'.format(K))
    # ax.legend()
    # fig.savefig("knn_experiment2/{}_K{}.pdf".format(dataset_name, K))

        # knn_result_exp.add_knn_result(dataset_name, K, knn_method_name, parameter_name, parameter_list,
        #                               quality_name, quality_list, time_list)
    
        
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
        # knn_result_exp.add_knn_result(dataset_name, K, knn_method_name, parameter_name, parameter_list,
        #                               quality_name, quality_list, time_list)

    
    # if not skip_save:
    #     knn_result_exp.save()
    
    # if save_plot:
    #     knn_result_exp.plot(dataset_name, K, quality_name, dataX,
    #                         dash_method=["FLATL2"])