#ifndef __BUILD_TREE_UTILS__CU
#define __BUILD_TREE_UTILS__CU

__global__
void
build_tree_utils(int* actual_depth,
                 int* depth_level_count,
                 int* count_new_nodes,
                 int* tree_count,
                 int* accumulated_nodes_count,
                 int* child_count,
                 int* accumulated_child_count,
                 int* device_active_points_count)
// Executed at end of each iteration of tree building
{
    depth_level_count[*actual_depth] = *count_new_nodes;
    accumulated_nodes_count[*actual_depth] = accumulated_nodes_count[*actual_depth-1] + *count_new_nodes;
    
    *actual_depth = *actual_depth+1;
    *count_new_nodes = 0;
    *tree_count = 0;
    *device_active_points_count = 0;
}


__global__
void
build_tree_fix(int* depth_level_count,
               int* tree_count,
               int* accumulated_nodes_count,
               int max_depth){
    *tree_count = 0;
    for(int i=0; i < max_depth; ++i){
        accumulated_nodes_count[i] = *tree_count;
        *tree_count += depth_level_count[i];
    }
}

__global__
void
build_tree_max_leaf_size(int* max_leaf_size,
                         int* min_leaf_size,
                         int* total_leafs,
                         bool* is_leaf,
                         int* child_count,
                         int* count_nodes,
                         int* depth_level_count,
                         int depth)
{
    int tid = blockDim.x*blockIdx.x+threadIdx.x;

    for(int node=tid; node < depth_level_count[depth]; node+=blockDim.x*gridDim.x){
        if(is_leaf[node]){
            atomicMax(max_leaf_size, child_count[node]);
            atomicMin(min_leaf_size, child_count[node]);
            atomicAdd(total_leafs, 1);
        }
    }
}

__global__
void
build_tree_set_leafs_idx(int* leaf_idx_to_node_idx,
                         int* node_idx_to_leaf_idx,
                         int* total_leafs,
                         bool* is_leaf,
                         int* depth_level_count,
                         int* accumulated_nodes_count,
                         int depth)
{
    int tid = blockDim.x*blockIdx.x+threadIdx.x;
    int new_idx;
    int acc_idx;
    for(int node=tid; node < depth_level_count[depth]; node+=blockDim.x*gridDim.x){
        if(is_leaf[node]){
            acc_idx = accumulated_nodes_count[depth] + node;
            new_idx = atomicAdd(total_leafs, 1);
            leaf_idx_to_node_idx[new_idx] = acc_idx;
            node_idx_to_leaf_idx[acc_idx] = new_idx;
        }
    }
}

#endif