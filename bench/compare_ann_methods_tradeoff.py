import time
import faiss
# Load MNIST from Scikit-learn
import numpy as np
# argparse
import argparse
import logging

from utils.metrics import get_projection, get_nne_rate
from utils.result_manager import KnnResult
from utils.datasets import load_dataset_knn, load_dataset

# import PCA
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def get_gpu_rsfk_results(
    options
    ):
    from gpu_rsfk.RSFK import RSFK

    rsfk = RSFK(random_state=0)
    # indices, dist = rsfk.find_nearest_neighbors(X,
    #                                             options.k_neighbors,
    #                                             nn_exploring_factor=False,
    #                                             verbose=options.verbose,
    #                                             n_trees=10)  # number of trees
    # parameter_list = [1, 5, 10, 20, 30, 100, 200]
    # parameter_list = [1, 5, 10, 20, 30, 40, 50]
    # parameter_list = [1, 5, 10]
    parameter_list = [1, 2, 3, 4, 5, 8, 16]
    # parameter_list = [1, 2, 4]
    # parameter_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 40, 50]
    # parameter_list = [1, 2, 4, 8, 16, 32, 64, 128]
    # parameter_list = [1]
    # parameter_list = [4]
    knn_method_name = "ANN-RSFK"
    kr = KnnResult(
        RSFK,
        options.dataset,
        options.k_neighbors,
        dataset_id=options.dataset_id,
        n_points=options.n_samples,
        ndim=options.n_features,
        dir_path=".",
        parameter_name='n_trees',
        experiment_name="exp",
        quality_metric='nnp_rate',
        model_initial_params={},
        model_find_params=dict(
            # add_bit_random_motion=True,
            # random_motion_force=0.1,
            ensure_valid_indices=False,
            min_tree_children=32,
            # min_tree_children=256,
            # max_tree_children=128,
            max_tree_children=512,
            # max_tree_children=1024,
            # max_tree_children=2048,
            # max_tree_depth=5000,

            # random_motion_force=0.01,
            # nn_exploring_factor=2,
            nn_exploring_factor=False,

            # verbose=0,
            verbose=1,
            # verbose=2,
            # partition_method='kmeans',
            # partition_method=None,
            # partition_method='random+kmeans',
        ),
        save_after_add=False)
    
    mname = knn_method_name +f" (KMeans-(B{kr.model_find_params['max_tree_children']}) Partitioning)"
    kr.evaluate_parameter_list(
        parameter_list,
        partition_method='kmeans',
        model_name=mname,
        ).clean()

    mname = knn_method_name +f" (Random Partitioning-(B{kr.model_find_params['max_tree_children']})"
    kr.evaluate_parameter_list(
        parameter_list,
        partition_method='random',
        model_name=mname,
        ).clean()

    mname = knn_method_name +f" (Random+KMeans Partitioning-(B{kr.model_find_params['max_tree_children']})"
    kr.evaluate_parameter_list(
        parameter_list,
        partition_method='random+kmeans',
        model_name=mname,
        ).clean()
    kr.save()

    # kr.plot(
    #     [options.dataset],
    #     options.k_neighbors,
    #     'nnp_rate',
    #     # baseline="Brute Force",
    # )
    return kr

def get_faiss_ivfflat_results(
    options
    ):
    knn_method_name = "IVFFLAT"
    parameter_name = "nprobe"
    parameter_list = [x+1 for x in range(20)]

    kr = KnnResult(
        faiss.IndexIVFFlat,
        options.dataset,
        options.k_neighbors,
        dataset_id=options.dataset_id,
        n_points=options.n_samples,
        ndim=options.n_features,
        dir_path=".",
        parameter_name=parameter_name,
        experiment_name="exp",
        quality_metric='nnp_rate',
        model_initial_params={},
        model_find_params={},
        save_after_add=False)
    # parameter_list = [20]
    kr.evaluate_parameter_list(
        parameter_list,
        model_name=knn_method_name,
        requires_model_inst=False
        ).clean()
    kr.save()
    # kr.plot(
    #     [options.dataset],
    #     options.k_neighbors,
    #     'nnp_rate',
    #     baseline="IVFFLAT",
    # )
    return kr
    quality_list = []
    time_list = []
    
    dataX, y = load_dataset(
        options.dataset,
        random_seed=0,
        npoints=options.n_samples,
        ndim=options.n_features
    )
    d = dataX.shape[1]
    res = faiss.StandardGpuResources()  # use a single GPU
    xb = np.require(dataX, np.float32, ['CONTIGUOUS', 'ALIGNED'])
    xq = np.require(dataX, np.float32, ['CONTIGUOUS', 'ALIGNED'])
    real_distances, real_indices, brute_force_time = load_dataset_knn(
                    options.dataset,
                    max_k=options.k_neighbors,
                    npoints=options.n_samples,
                    ndim=options.n_features,
                    return_brute_force_time=True
                )
    for nprobe in parameter_list:
        init_t = time.time()
        quantizer = faiss.IndexFlatL2(d)  # the other index
        index = faiss.IndexIVFFlat(quantizer, d, 256, faiss.METRIC_L2)
        # index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
        
        index = faiss.index_cpu_to_gpu(res, 0, index)

        index.nprobe = nprobe              # default nprobe is 1, try a few more
        # here we specify METRIC_L2, by default it performs inner-product search
        assert not index.is_trained
        index.train(xb)
        assert index.is_trained
        

        index.add(xb)                  # add may be a bit slower as well
        
        # index = faiss.index_cpu_to_gpu(res, 0, index)
        D, I = index.search(xq, options.k_neighbors)     # actual search

        t = time.time() - init_t
        # nne_rate = quality_function(real_indices, I, real_sqd_dist, D, max_k=K)
        nne_rate = quality = get_nne_rate(
                real_indices,
                I,
                random_state=0,
                max_k=options.k_neighbors,
                verbose=0
            )
        

        print("FAISS IVFFLAT (nprobe={}) takes {} seconds".format(nprobe,t))
        print("FAISS IVFFLAT NNE (nprobe={}): {}".format(nprobe,nne_rate))
        quality_list.append(nne_rate)
        time_list.append(t)
    
    knn_result_exp = KnnResult(
        None,
        options.dataset,
        options.k_neighbors,
        dataset_id=options.dataset_id,
        n_points=options.n_samples,
        ndim=options.n_features,
        dir_path=".",
        parameter_name=parameter_name,
        experiment_name="faiss_ivfflat_exp",
        quality_metric='nnp_rate',
        model_initial_params={},
        model_find_params={},
        save_after_add=False)
    
    K = options.k_neighbors
    dataset_name = options.dataset
    knn_result_exp.add_knn_result(
        dataset_name,
        K,
        knn_method_name,
        parameter_name,
        parameter_list,
        "nnp_rate",
        quality_list,
        time_list)
    knn_result_exp.save()
    knn_result_exp.plot(
        [options.dataset],
        options.k_neighbors,
        'nnp_rate',
        # baseline="Brute Force",
    )

    
def main():
    parser = argparse.ArgumentParser(description="Load MNIST or artificial dataset")
    parser.add_argument('--dataset', type=str, default='MNIST',
                        help='Dataset to load (default: MNIST). Supported: MNIST, ARTIFICIAL_UNIFORM, KDDCUP99 or any other dataset available in when calling sklearn.datasets.fetch_openml().',
                        # choices=['MNIST', 'KDDCUP99', 'ARTIFICIAL_UNIFORM',]
                        )
    parser.add_argument('--dataset_id', type=int, default=None,
                        help='Dataset ID from OpenML to load (overrides --dataset if provided).')
    parser.add_argument('--n_samples', type=int, default=10000,
                        help='Number of samples for artificial dataset (default: 10000)')
    parser.add_argument('--n_features', type=int, default=128,
                        help='Number of features for artificial dataset (default: 128)')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')
    # K
    parser.add_argument('-k', '--k_neighbors', type=int, default=32,
                        help='Number of neighbors to search for (default: 32)')
    parser.add_argument('--run_faiss_ivfflat', action='store_true',
                        help='Run Faiss IVFFLAT experiments instead of RSFK experiments')
    parser.add_argument('--run_rsfk', action='store_true',
                        help='Run RSFK experiments (default: True)')
    
    args = parser.parse_args()

    k_neighbors = args.k_neighbors

    if args.verbose:
        logger.info("Starting RSFK experiments...")

    kr = None
    if args.run_faiss_ivfflat:
        kr = get_faiss_ivfflat_results(args)
    if args.run_rsfk:
        kr = get_gpu_rsfk_results(args)

    if kr is None:
        kr = KnnResult(
            None,
            args.dataset,
            args.k_neighbors,
            dataset_id=args.dataset_id,
            n_points=args.n_samples,
            ndim=args.n_features,
            dir_path=".",
            parameter_name='n_trees',
            experiment_name="exp",
            quality_metric='nnp_rate',
            model_initial_params={},
            model_find_params={},
            save_after_add=False)
    logger.info("RSFK results obtained.")
    if kr:
        if args.verbose:
            logger.info("Results summary:")
            kr.print_summary()

        kr.plot(
            [args.dataset],
            args.k_neighbors,
            'nnp_rate',
            dash_method=["Brute Force"],
            baseline="IVFFLAT",
        )
    
    


if __name__ == "__main__":
    main()