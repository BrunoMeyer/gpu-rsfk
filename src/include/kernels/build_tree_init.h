#ifndef __BUILD_TREE_INIT__H
#define __BUILD_TREE_INIT__H

#include "../../kernels/build_tree_init.cu"
#include "../../kernels/create_node.cu"

__global__
void build_tree_init(typepoints* tree,
                     int* tree_parents,
                     int* tree_children,
                     int* points_parent,
                     int* child_count,
                     bool* is_leaf,
                     typepoints* points,
                     int* actual_depth,
                     int* tree_count,
                     int* depth_level_count,
                     int* accumulated_nodes_count,
                     int* accumulated_child_count,
                     int* device_count_points_on_leafs,
                     int* count_new_nodes,
                     int N, int D, int RANDOM_SEED);

#endif