
// Implementation file for the python extensions

// #include "pymodule_ext.h"

#include <sys/types.h>
#include "../rpfk.cu"

extern "C" {
    void pymodule_rpfk_knn(int n_trees,
                           int K,
                           int N,
                           int D,
                           int MIN_TREE_CHILD,
                           int MAX_TREE_CHILD,
                           int MAX_DEPTH,
                           int VERBOSE,
                           int RANDOM_STATE,
                           int nn_exploring_factor,
                           RSFK_typepoints* points,
                           int* knn_indices,
                           RSFK_typepoints* knn_sqr_distances);

    void pymodule_cluster_by_sample_tree(int N,
                                         int D,
                                         int MIN_TREE_CHILD,
                                         int MAX_TREE_CHILD,
                                         int MAX_DEPTH,
                                         int VERBOSE,
                                         int RANDOM_STATE,
                                         RSFK_typepoints* points,
                                         int** nodes_buckets,
                                         int** bucket_sizes,
                                         int* total_leaves,
                                         int* max_child);
}

void pymodule_rpfk_knn(int n_trees,
                       int K,
                       int N,
                       int D,
                       int MIN_TREE_CHILD,
                       int MAX_TREE_CHILD,
                       int MAX_DEPTH,
                       int VERBOSE,
                       int RANDOM_STATE,
                       int nn_exploring_factor,
                       RSFK_typepoints* points,
                       int* knn_indices,
                       RSFK_typepoints* knn_sqr_distances)
{
    std::string run_name="run";
    RPFK rpfk_knn(points, knn_indices, knn_sqr_distances,
                  MIN_TREE_CHILD, MAX_TREE_CHILD,
                  MAX_DEPTH, RANDOM_STATE, nn_exploring_factor);

    rpfk_knn.knn_gpu_rpfk_forest(n_trees, K, N, D, VERBOSE, run_name);
}

void pymodule_cluster_by_sample_tree(int N,
                                     int D,
                                     int MIN_TREE_CHILD,
                                     int MAX_TREE_CHILD,
                                     int MAX_DEPTH,
                                     int VERBOSE,
                                     int RANDOM_STATE,
                                     RSFK_typepoints* points,
                                     int** nodes_buckets,
                                     int** bucket_sizes,
                                     int* total_leaves,
                                     int* max_child)
{
    std::string run_name="run";

    RPFK rpfk_knn(points, nullptr, nullptr,
                  MIN_TREE_CHILD, MAX_TREE_CHILD,
                  MAX_DEPTH, RANDOM_STATE, 0);

    TreeInfo tinfo;

    tinfo = rpfk_knn.cluster_by_sample_tree(N, D, VERBOSE,
                                            nodes_buckets,
                                            bucket_sizes,
                                            run_name);

    *total_leaves = tinfo.total_leaves;
    *max_child = tinfo.max_child;
}