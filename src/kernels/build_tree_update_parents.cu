#ifndef __BUILD_TREE_UPDATE_PARENTS__CU
#define __BUILD_TREE_UPDATE_PARENTS__CU

#include "../include/common.h"

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
                               int MIN_TREE_CHILD, int MAX_TREE_CHILD)
{
    int tid = blockDim.x*blockIdx.x+threadIdx.x;

    int right_child, p;
    int updated_count;
    // Set nodes parent in the new depth
    for(p = tid; p < N; p+=blockDim.x*gridDim.x){
        if(points_depth[p] == *actual_depth-1 && !is_leaf[points_parent[p]]){
            right_child = is_right_child[p];
            points_parent[p] = tree_children[2*points_parent[p]+right_child];
            points_depth[p] = *actual_depth;
            updated_count = atomicAdd(&child_count_new_depth[points_parent[p]],1)+1;

            if(updated_count > MAX_TREE_CHILD-2){
                is_leaf_new_depth[points_parent[p]] = false;
            }
        }
    }
}

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
                                    #if COMPILE_TYPE == DEBUG
                                    int* count_undo_leaf,
                                    #endif
                                    int N, int D,
                                    int MIN_TREE_CHILD, int MAX_TREE_CHILD)
{
    int tid = blockDim.x*blockIdx.x+threadIdx.x;

    int p, right_child;
    int parent_leaf_node;
    int new_count;
    // Set nodes parent in the new depth
    for(p = tid; p < N; p+=blockDim.x*gridDim.x){
        if(points_depth[p] == *actual_depth && child_count_new_depth[points_parent[p]] < MIN_TREE_CHILD){
            child_count_new_depth[points_parent[p]] = 0;
            is_leaf_new_depth[points_parent[p]] = false;
            
            // If the parent node was a left node then this point must be moved
            // to the right sibling . Otherwise it will be moved to the right sibling
            right_child = abs(is_right_child[p]-1);
            
            parent_leaf_node = 2*tree_parents_new_depth[points_parent[p]];
            
            points_parent[p] = tree_children[parent_leaf_node+right_child];

            is_leaf_new_depth[points_parent[p]] = true;
            new_count = atomicAdd(&child_count_new_depth[points_parent[p]],1)+1;
            // If at least one thread reach this (the ones related to id bigger than MAX_TREE_CHILD)
            // Then, this parent certainly are not a leaf.
            // This may lead to many overwriting in the same memory pointer, but
            // it will not affect the result
            if(new_count > MAX_TREE_CHILD){
                #if COMPILE_TYPE == DEBUG
                if(new_count-1 == MAX_TREE_CHILD) atomicAdd(count_undo_leaf,1);
                #endif
                is_leaf_new_depth[points_parent[p]] = false;
            }
        }
    }
}

#endif