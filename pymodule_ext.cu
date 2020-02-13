
// Implementation file for the python extensions

// #include "pymodule_ext.h"

#include <sys/types.h>
#include "rptk.cu"

extern "C" {
    void pymodule_rptk_knn(int n_trees,
                       int K,
                       int N,
                       int D,
                       int MAX_DEPTH,
                       int VERBOSE,
                       typepoints* points,
                       int* knn_indices,
                       typepoints* knn_sqr_distances);
}

void pymodule_rptk_knn(int n_trees,
                       int K,
                       int N,
                       int D,
                       int MAX_DEPTH,
                       int VERBOSE,
                       typepoints* points,
                       int* knn_indices,
                       typepoints* knn_sqr_distances)
{
    string run_name="run";
    RPTK rptk_knn(points, knn_indices, knn_sqr_distances);

    rptk_knn.knn_gpu_rptk_forest(n_trees, K, N, D, MAX_DEPTH, VERBOSE, run_name);
}