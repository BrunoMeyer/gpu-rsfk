#ifndef __NEAREST_NEIGHBORS_EXPLORING__H
#define __NEAREST_NEIGHBORS_EXPLORING__H

#include "../../kernels/nearest_neighbors_exploring.cu"

__global__
void nearest_neighbors_exploring(typepoints* points,
                                 int* old_knn_indices,
                                 int* knn_indices,
                                 typepoints* knn_sqr_dist,
                                 int N, int D, int K);

#endif