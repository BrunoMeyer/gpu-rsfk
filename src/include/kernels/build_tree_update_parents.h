#ifndef __BUILD_TREE_UPDATE_PARENTS__H
#define __BUILD_TREE_UPDATE_PARENTS__H

#include "../../kernels/build_tree_update_parents.cu"

__global__
void build_tree_update_parents(typepoints* tree,
                               int* tree_parents,
                               int* tree_children,
                               int* points_parent,
                               int* points_depth,
                               int* is_right_child,
                               bool* is_leaf,
                               bool* is_leaf_new_depth,
                               int* child_count,
                               int* child_count_new_depth,
                               typepoints* points,
                               int* actual_depth,
                               int* tree_count,
                               int* depth_level_count,
                               int* count_new_nodes,
                               int N, int D,
                               int MIN_TREE_CHILD, int MAX_TREE_CHILD);

__global__
void build_tree_post_update_parents(typepoints* tree,
                                    int* tree_parents_new_depth,
                                    int* tree_children,
                                    int* points_parent,
                                    int* points_depth,
                                    int* is_right_child,
                                    bool* is_leaf,
                                    bool* is_leaf_new_depth,
                                    int* child_count,
                                    int* child_count_new_depth,
                                    typepoints* points,
                                    int* actual_depth,
                                    int* tree_count,
                                    int* depth_level_count,
                                    int* count_new_nodes,
                                    #if COMPILE_TYPE == COMPILE_TYPE_DEBUG
                                    int* count_undo_leaf,
                                    #endif
                                    int N, int D,
                                    int MIN_TREE_CHILD, int MAX_TREE_CHILD);

#endif