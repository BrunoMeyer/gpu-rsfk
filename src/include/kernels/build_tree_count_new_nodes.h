#ifndef __BUILD_TREE_COUNT_NEW_NODES__H
#define __BUILD_TREE_COUNT_NEW_NODES__H
#include "../common.h"
#include "../../kernels/build_tree_count_new_nodes.cu"

__global__
void build_tree_count_new_nodes(RSFK_typepoints* tree,
                                int* tree_parents,
                                int* tree_children,
                                int* points_parent,
                                int* is_right_child,
                                bool* is_leaf,
                                int* count_points_on_leafs,
                                int* child_count,
                                RSFK_typepoints* points,
                                int* actual_depth,
                                int* tree_count,
                                int* depth_level_count,
                                int* count_new_nodes,
                                int N, int D,
                                int MIN_TREE_CHILD, int MAX_TREE_CHILD);


__global__
void build_tree_accumulate_child_count(int* depth_level_count,
                                       int* count_new_nodes,
                                       int* count_points_on_leafs,
                                       int* accumulated_child_count,
                                       int* actual_depth);

__global__
void build_tree_organize_sample_candidates(int* points_parent,
                                           int* points_depth,
                                           bool* is_leaf,
                                           int* is_right_child,
                                           int* count_new_nodes,
                                           int* count_points_on_leafs,
                                           int* accumulated_child_count,
                                           int* sample_candidate_points,
                                           int* points_id_on_sample,
                                           int* actual_depth,
                                           int N);

#endif