
// Implementation file for the python extensions

// #include "pymodule_ext.h"

#include <sys/types.h>
#include "rptk.cu"

extern "C" {
    void pymodule_rptk_knn(int n_trees,
                       int K,
                       int N,
                       int D,
                       int MAX_TREE_CHILD,
                       int MAX_DEPTH,
                       int VERBOSE,
                       int RANDOM_STATE,
                       int nn_exploring_factor,
                       typepoints* points,
                       int* knn_indices,
                       typepoints* knn_sqr_distances);
}

void pymodule_rptk_knn(int n_trees,
                       int K,
                       int N,
                       int D,
                       int MAX_TREE_CHILD,
                       int MAX_DEPTH,
                       int VERBOSE,
                       int RANDOM_STATE,
                       int nn_exploring_factor,
                       typepoints* points,
                       int* knn_indices,
                       typepoints* knn_sqr_distances)
{
    string run_name="run";
    RPTK rptk_knn(points, knn_indices, knn_sqr_distances, MAX_TREE_CHILD,
                  MAX_DEPTH, RANDOM_STATE, nn_exploring_factor);

    rptk_knn.knn_gpu_rptk_forest(n_trees, K, N, D, VERBOSE, run_name);
}