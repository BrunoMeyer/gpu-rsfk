#ifndef __BUILD_TREE_BUCKET_POINTS__CU
#define __BUILD_TREE_BUCKET_POINTS__CU

__global__
void build_tree_bucket_points(int* points_parent,
                              int* points_depth,
                              int* accumulated_nodes_count,
                              int* bucket_nodes,
                              int* tmp_node_child_count,
                              int N, int bucket_size)
{
    int tid = blockDim.x*blockIdx.x+threadIdx.x;
    // int right_child, p;
    int my_id_on_bucket, parent_id;
    for(int p = tid; p < N; p+=blockDim.x*gridDim.x){
        parent_id = accumulated_nodes_count[points_depth[p]] + points_parent[p];
        my_id_on_bucket = atomicAdd(&tmp_node_child_count[parent_id], 1);
        bucket_nodes[parent_id*bucket_size + my_id_on_bucket] = p;
        
        // printf("%d %d %d | %d\n", parent_id, bucket_size, my_id_on_bucket, parent_id*bucket_size + my_id_on_bucket);
        // if(!is_leaf[points_parent[p]]){
        //     right_child = check_hyperplane_side(points_parent[p], p, tree, points, D);
        //     points_parent[p] = HEAP_LEFT(points_parent[p])+right_child;
        // }
        // atomicAdd(&device_child_count[points_parent[p]],1);
        // device_sample_points[2*points_parent[p] + curand(&r) % 2] = p;
    }
}

#endif