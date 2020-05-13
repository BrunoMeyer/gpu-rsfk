#ifndef __BUILD_TREE_BUCKET_POINTS__H
#define __BUILD_TREE_BUCKET_POINTS__H

#include "../common.h"
#include "../../kernels/build_tree_bucket_points.cu"

__global__
void build_tree_bucket_points(int* points_parent,
                              int* points_depth,
                              int* accumulated_nodes_count,
                              int* node_idx_to_leaf_idx,
                              int* nodes_bucket,
                              int* bucket_sizes,
                              int N, int max_bucket_size, int total_leafs);
#endif