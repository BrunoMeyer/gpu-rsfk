#ifndef __BUILD_TREE_UTILS__H
#define __BUILD_TREE_UTILS__H

#include "../../kernels/build_tree_utils.cu"

__global__
void
build_tree_utils(int* actual_depth,
                 int* depth_level_count,
                 int* count_new_nodes,
                 int* tree_count,
                 int* accumulated_nodes_count,
                 int* child_count,
                 int* accumulated_child_count,
                 int* device_active_points_count);


__global__
void
build_tree_fix(int* depth_level_count,
               int* tree_count,
               int* accumulated_nodes_count,
               int max_depth);

__global__
void
build_tree_max_leaf_size(int* max_leaf_size,
                         int* min_leaf_size,
                         int* total_leafs,
                         bool* is_leaf,
                         int* child_count,
                         int* count_nodes,
                         int* depth_level_count,
                         int depth);

__global__
void
build_tree_set_leaves_idx(int* leaf_idx_to_node_idx,
                         int* node_idx_to_leaf_idx,
                         int* total_leaves,
                         bool* is_leaf,
                         int* depth_level_count,
                         int* accumulated_nodes_count,
                         int depth);

__global__
void
debug_count_hist_leaf_size(int* hist,
                           int* max_leaf_size,
                           int* min_leaf_size,
                           int* total_leafs,
                           bool* is_leaf,
                           int* child_count,
                           int* count_nodes,
                           int* depth_level_count,
                           int depth);

#endif