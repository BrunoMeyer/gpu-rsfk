#ifndef __BUILD_TREE_CREATE_NODES__H
#define __BUILD_TREE_CREATE_NODES__H

#include "../../kernels/create_node.cu"
#include "../../kernels/build_tree_create_nodes.cu"

__device__
inline
void create_root(RSFK_typepoints* tree,
                 int* tree_parents,
                 int* tree_children,
                 int* tree_count,
                 int p1,
                 int p2,
                 int* count_new_nodes,
                 RSFK_typepoints* points,
                 int D, int N);

__device__
inline
void create_node(int parent,
                 int is_right_child,
                 RSFK_typepoints* tree,
                 int* tree_parents,
                 int* tree_children,
                 int* tree_count,
                 int* count_new_nodes,
                 int p1,
                 int p2,
                 RSFK_typepoints* points,
                 int D, int N);

__global__
void build_tree_create_nodes(RSFK_typepoints* tree_new_depth,
                             int* tree_parents_new_depth,
                             int* tree_children,
                             int* points_parent,
                             int* points_depth,
                             int* is_right_child,
                             bool* is_leaf,
                             bool* is_leaf_new_depth,
                             int* child_count,
                             RSFK_typepoints* points,
                             int* actual_depth,
                             int* tree_count,
                             int* depth_level_count,
                             int* count_new_nodes,
                             int* accumulated_nodes_count,
                             int* accumulated_child_count,
                             int* count_points_on_leafs,
                             int* sample_candidate_points,
                             int N, int D, int MIN_TREE_CHILD,
                             int MAX_TREE_CHILD, int RANDOM_SEED);



/*
__global__
void init_random_directions(RSFK_typepoints* random_directions,
                            int random_directions_size,
                            int* actual_depth,
                            int RANDOM_SEED);

__global__
void project_active_points(RSFK_typepoints* projection_values,
                           RSFK_typepoints* points,
                           int* active_points,
                           int* active_points_count,
                           int* points_parent,
                           int* is_right_child,
                           int* sample_candidate_points,
                           RSFK_typepoints* min_random_proj_values,
                           RSFK_typepoints* max_random_proj_values,
                           int N, int D);

__global__
void choose_points_to_split(RSFK_typepoints* projection_values,
                           int* points_parent,
                           int* active_points,
                           int* active_points_count,
                           int* is_right_child,
                           int* sample_candidate_points,
                           RSFK_typepoints* min_random_proj_values,
                           RSFK_typepoints* max_random_proj_values,
                           int N, int D);

__global__
void build_tree_create_nodes_random_projection(RSFK_typepoints* tree_new_depth,
                             int* tree_parents_new_depth,
                             int* tree_children,
                             int* points_parent,
                             int* points_depth,
                             int* is_right_child,
                             bool* is_leaf,
                             bool* is_leaf_new_depth,
                             int* child_count,
                             RSFK_typepoints* points,
                             int* actual_depth,
                             int* tree_count,
                             int* depth_level_count,
                             int* count_new_nodes,
                             int* accumulated_nodes_count,
                             int* accumulated_child_count,
                             int* count_points_on_leafs,
                             int* sample_candidate_points,
                             int* random_directions,
                             int N, int D, int K, int MAX_TREE_CHILD, int RANDOM_SEED);
*/
#endif