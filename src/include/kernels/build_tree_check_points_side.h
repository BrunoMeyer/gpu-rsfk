#ifndef __BUILD_TREE_CHECK_POINTS_SIDE__H
#define __BUILD_TREE_CHECK_POINTS_SIDE__H

// #include "../common.h"
#include "../../kernels/build_tree_check_points_side.cu"

__device__
inline
int check_hyperplane_side(int node_idx,
                          int p,
                          typepoints* tree,
                          typepoints* points,
                          int D,
                          int N,
                          int* count_new_nodes);


__global__
void build_tree_check_active_points(int* points_parent,
                                    int* points_depth,
                                    bool* is_leaf,
                                    int* actual_depth,
                                    int* active_points,
                                    int* active_points_count,
                                    int N);


__global__
void build_tree_check_points_side(typepoints* tree,
                                  int* tree_parents,
                                  int* tree_children,
                                  int* points_parent,
                                  int* points_depth,
                                  int* is_right_child,
                                  bool* is_leaf,
                                  int* child_count,
                                  typepoints* points,
                                  int* actual_depth,
                                  int* tree_count,
                                  int* depth_level_count,
                                  int* accumulated_child_count,
                                  int* count_points_on_leafs,
                                  int* sample_candidate_points,
                                  int* points_id_on_sample,
                                  int* active_points,
                                  int* active_points_count,
                                  int* count_new_nodes,
                                  int N, int D, int RANDOM_SEED);

#endif