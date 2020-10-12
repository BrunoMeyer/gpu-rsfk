
// Implementation file for the python extensions

// #include "pymodule_ext.h"

#include <sys/types.h>
#include "../rpfk.cu"
#include <thrust/fill.h>

extern "C" {
    void pymodule_rpfk_knn(int n_trees,
                           int num_neighbors,
                           int N,
                           int D,
                           int MIN_TREE_CHILD,
                           int MAX_TREE_CHILD,
                           int MAX_DEPTH,
                           int VERBOSE,
                           int RANDOM_STATE,
                           int nn_exploring_factor,
                           typepoints* points,
                           int* knn_indices,
                           typepoints* knn_sqr_distances);

    void pymodule_cluster_by_sample_tree(int N,
                                         int D,
                                         int MIN_TREE_CHILD,
                                         int MAX_TREE_CHILD,
                                         int MAX_DEPTH,
                                         int VERBOSE,
                                         int RANDOM_STATE,
                                         typepoints* points,
                                         int** nodes_buckets,
                                         int** bucket_sizes,
                                         int* total_leaves,
                                         int* max_child);

    void pymodule_create_cluster_with_hbgf(int n_trees,
                                           int K,
                                           int n_eig_vects,
                                           int N,
                                           int D,
                                           int MIN_TREE_CHILD,
                                           int MAX_TREE_CHILD,
                                           int MAX_DEPTH,
                                           int VERBOSE,
                                           int RANDOM_STATE,
                                           typepoints* points,
                                           int* result);

    void pymodule_spectral_clustering_with_knngraph(int n_trees,
                                                    int K, //number of clusters
                                                    int num_neighbors,
                                                    int nn_exploring_factor,
                                                    int n_eig_vects,
                                                    int N,
                                                    int D,
                                                    int MIN_TREE_CHILD,
                                                    int MAX_TREE_CHILD,
                                                    int MAX_DEPTH,
                                                    int VERBOSE,
                                                    int RANDOM_STATE,
                                                    typepoints* points,
                                                    int* result);
}

void pymodule_rpfk_knn(int n_trees,
                       int num_neighbors,
                       int N,
                       int D,
                       int MIN_TREE_CHILD,
                       int MAX_TREE_CHILD,
                       int MAX_DEPTH,
                       int VERBOSE,
                       int RANDOM_STATE,
                       int nn_exploring_factor,
                       typepoints* points,
                       int* knn_indices,
                       typepoints* knn_sqr_distances)
{
    std::string run_name="run";
    RPFK rpfk_knn(points, knn_indices, knn_sqr_distances,
                  MIN_TREE_CHILD, MAX_TREE_CHILD,
                  MAX_DEPTH, RANDOM_STATE, nn_exploring_factor);

    rpfk_knn.knn_gpu_rpfk_forest(n_trees, num_neighbors, N, D, VERBOSE, run_name);
}

void pymodule_cluster_by_sample_tree(int N,
                                     int D,
                                     int MIN_TREE_CHILD,
                                     int MAX_TREE_CHILD,
                                     int MAX_DEPTH,
                                     int VERBOSE,
                                     int RANDOM_STATE,
                                     typepoints* points,
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


void pymodule_create_cluster_with_hbgf(int n_trees,
                                       int K,
                                       int n_eig_vects,
                                       int N,
                                       int D,
                                       int MIN_TREE_CHILD,
                                       int MAX_TREE_CHILD,
                                       int MAX_DEPTH,
                                       int VERBOSE,
                                       int RANDOM_STATE,
                                       typepoints* points,
                                       int* result)
{
    std::string run_name="run";

    RPFK rpfk_knn(points, nullptr, nullptr,
                  MIN_TREE_CHILD, MAX_TREE_CHILD,
                  MAX_DEPTH, RANDOM_STATE, 0);

    int err;
    err = rpfk_knn.create_cluster_with_hbgf(result,
                                            n_trees,
                                            N, D, VERBOSE,
                                            K, n_eig_vects,
                                            run_name);

}


void pymodule_spectral_clustering_with_knngraph(int n_trees,
                                                int K, //number of clusters
                                                int num_neighbors,
                                                int nn_exploring_factor,
                                                int n_eig_vects,
                                                int N,
                                                int D,
                                                int MIN_TREE_CHILD,
                                                int MAX_TREE_CHILD,
                                                int MAX_DEPTH,
                                                int VERBOSE,
                                                int RANDOM_STATE,
                                                typepoints* points,
                                                int* result)
{
    std::string run_name="run";

    int* knn_indices     = new int[N*num_neighbors];
    float* knn_distances = new float[N*num_neighbors];
    thrust::fill(knn_indices, knn_indices+ N*num_neighbors, -1);
    thrust::fill(knn_distances, knn_distances+ N*num_neighbors, FLT_MAX);

    RPFK rpfk_knn(points, knn_indices, knn_distances,
                  MIN_TREE_CHILD, MAX_TREE_CHILD,
                  MAX_DEPTH, RANDOM_STATE, nn_exploring_factor);

    rpfk_knn.knn_gpu_rpfk_forest(n_trees, num_neighbors,
                                 N, D, VERBOSE, run_name);

    int err;
    err = rpfk_knn.spectral_clustering_with_knngraph(result,
                                                     num_neighbors,
                                                     N, D, VERBOSE,
                                                     K, n_eig_vects,
                                                     true,
                                                     run_name);
    // delete [] knn_indices;
    delete [] knn_distances;
}