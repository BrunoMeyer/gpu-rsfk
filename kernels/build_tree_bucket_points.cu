#ifndef __BUILD_TREE_BUCKET_POINTS__CU
#define __BUILD_TREE_BUCKET_POINTS__CU

__global__
void build_tree_bucket_points(int* points_parent,
                              int* points_depth,
                              int* accumulated_nodes_count,
                              int* bucket_nodes,
                              int* count_buckets,
                              int N, int bucket_size)
{
    int tid = blockDim.x*blockIdx.x+threadIdx.x;
    int my_id_on_bucket, parent_id;
    for(int p = tid; p < N; p+=blockDim.x*gridDim.x){
        parent_id = accumulated_nodes_count[points_depth[p]] + points_parent[p];
        // printf("%d\n", parent_id);
        my_id_on_bucket = atomicAdd(&count_buckets[parent_id], 1);
        bucket_nodes[parent_id*bucket_size + my_id_on_bucket] = p;
    }
}

#endif