#ifndef __BUILD_TREE_UTILS__CU
#define __BUILD_TREE_UTILS__CU

__global__
void
build_tree_utils(int* actual_depth, int* depth_level_count, int* count_new_nodes, int* tree_count){
    depth_level_count[*actual_depth] = *count_new_nodes;
    *actual_depth = *actual_depth+1;
    // depth_level_count[*actual_depth] = depth_level_count[*actual_depth - 1] + *count_new_nodes;
    *count_new_nodes = 0;
    *tree_count = 0;
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
                         int* total_leafs,
                         bool* is_leaf,
                         int* child_count,
                         int* count_nodes,
                         int* depth_level_count,
                         int depth)
{
    // __shared__ int s_max_leaf_size;
    int tid = blockDim.x*blockIdx.x+threadIdx.x;

    // if(threadIdx.x == 0){
    //     s_max_leaf_size = 0;
    // }
    // __syncthreads();
    for(int node=tid; node < depth_level_count[depth]; node+=blockDim.x*gridDim.x){
        if(is_leaf[node]){
            // atomicMax(&s_max_leaf_size, device_child_count[tid]);
            atomicMax(max_leaf_size, child_count[node]);
            atomicAdd(total_leafs, 1);
            // printf("DEBUG6: %d\n", child_count[node]);
        }
    }
    // __syncthreads();
    // if(threadIdx.x == 0){
    //     atomicMax(max_leaf_size, s_max_leaf_size);
    // }
}

#endif