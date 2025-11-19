# Load MNIST from Scikit-learn
from sklearn.datasets import fetch_openml
import numpy as np
# argparse
import argparse
import logging

from utils.metrics import get_projection
from utils.result_manager import KnnResult
from utils.datasets import load_dataset_knn, load_dataset

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def get_gpu_rsfk_results(options):
    from gpu_rsfk.RSFK import RSFK

    rsfk = RSFK(random_state=0)
    # indices, dist = rsfk.find_nearest_neighbors(X,
    #                                             options.k_neighbors,
    #                                             nn_exploring_factor=False,
    #                                             verbose=options.verbose,
    #                                             n_trees=10)  # number of trees
    # parameter_list = [1, 5, 10, 20, 30, 100, 200]
    parameter_list = [1, 5, 10, 20, 30, 40, 50]
    # parameter_list = [1, 5, 10]
    # parameter_list = [1]
    # parameter_list = [4]
    knn_method_name = "ANN-RSFK"
    kr = KnnResult(
        RSFK,
        options.dataset,
        options.k_neighbors,
        n_points=options.n_samples,
        ndim=options.n_features,
        dir_path=".",
        parameter_name='n_trees',
        experiment_name="exp",
        quality_metric='nnp_rate',
        model_initial_params={},
        model_find_params=dict(
            add_bit_random_motion=False,
            ensure_valid_indices=False,
            min_tree_children=32,
            max_tree_children=1024,
            # max_tree_depth=5000,

            # random_motion_force=0.1,
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
    
    kr.evaluate_parameter_list(
        parameter_list,
        partition_method='random',
        model_name=knn_method_name +" (Random Partitioning)"
        ).clean()

    kr.evaluate_parameter_list(
        parameter_list,
        partition_method='kmeans',
        model_name=knn_method_name +" (KMeans Partitioning)"
        ).clean()

    kr.evaluate_parameter_list(
        parameter_list,
        partition_method='random+kmeans',
        model_name=knn_method_name +" (Random+KMeans Partitioning)"
        ).clean()
    kr.save()

    kr.plot(
        [options.dataset],
        options.k_neighbors,
        'nnp_rate'
    )
    return kr

def main():
    parser = argparse.ArgumentParser(description="Load MNIST or artificial dataset")
    parser.add_argument('--dataset', type=str, default='MNIST',
                        help='Dataset to load (default: MNIST). Supported: MNIST, ARTIFICIAL_UNIFORM', choices=['MNIST', 'ARTIFICIAL_UNIFORM'])
    parser.add_argument('--n_samples', type=int, default=10000,
                        help='Number of samples for artificial dataset (default: 10000)')
    parser.add_argument('--n_features', type=int, default=128,
                        help='Number of features for artificial dataset (default: 128)')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')
    # K
    parser.add_argument('-k', '--k_neighbors', type=int, default=32,
                        help='Number of neighbors to search for (default: 32)')
    args = parser.parse_args()

    k_neighbors = args.k_neighbors

    if args.verbose:
        logger.info("Starting RSFK experiments...")
    rk = rsfk_results = get_gpu_rsfk_results(args)
    logger.info("RSFK results obtained.")
    if args.verbose:
        logger.info("Results summary:")
        rk.print_summary()


if __name__ == "__main__":
    main()