#ifndef __BUILD_TREE_BUCKET_POINTS__CU
#define __BUILD_TREE_BUCKET_POINTS__CU

__global__
void build_tree_bucket_points(int* points_parent,
                              int* points_depth,
                              int* accumulated_nodes_count,
                              int* node_idx_to_leaf_idx,
                              int* nodes_bucket,
                              int* bucket_sizes,
                              int N, int max_bucket_size, int total_leafs)
{
    int tid = blockDim.x*blockIdx.x+threadIdx.x;
    int my_id_on_bucket, parent_id;
    for(int p = tid; p < N; p+=blockDim.x*gridDim.x){
        parent_id = accumulated_nodes_count[points_depth[p]] + points_parent[p];
        my_id_on_bucket = atomicAdd(&bucket_sizes[node_idx_to_leaf_idx[parent_id]], 1);
        nodes_bucket[node_idx_to_leaf_idx[parent_id]*max_bucket_size + my_id_on_bucket] = p;
    }
}

#endif