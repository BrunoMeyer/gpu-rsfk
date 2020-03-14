#ifndef __BUILD_TREE_CREATE_NODES__CU
#define __BUILD_TREE_CREATE_NODES__CU

#include "create_node.cu"

__global__
void build_tree_create_nodes(typepoints* tree_new_depth,
                             int* tree_parents_new_depth,
                             int* tree_children,
                             int* points_parent,
                             int* points_depth,
                             int* is_right_child,
                             bool* is_leaf,
                             bool* is_leaf_new_depth,
                             int* child_count,
                             typepoints* points,
                             int* actual_depth,
                             int* tree_count,
                             int* depth_level_count,
                             int* count_new_nodes,
                             int* accumulated_nodes_count,
                             int* accumulated_child_count,
                             int* count_points_on_leafs,
                             int* sample_candidate_points,
                             int N, int D, int K, int MAX_TREE_CHILD, int RANDOM_SEED)
{
    int tid = blockDim.x*blockIdx.x+threadIdx.x;
    curandState_t r; 
    curand_init(*actual_depth*(RANDOM_SEED+blockDim.x)+RANDOM_SEED+tid, // the seed controls the sequence of random values that are produced
            blockIdx.x,  // the sequence number is only important with multiple cores 
            tid,  // the offset is how much extra we advance in the sequence for each call, can be 0 
            &r);

    int p1, p2, node_thread;
    // int parent_id;
    int rand_id;

    // Create new nodes
    for(node_thread = tid; node_thread < depth_level_count[*actual_depth-1]; node_thread+=blockDim.x*gridDim.x){
        for(int is_right=0; is_right < 2; ++is_right){
            if(!is_leaf[node_thread]){
                if(count_points_on_leafs[2*node_thread+is_right] > 0){
                    rand_id = (curand(&r) % count_points_on_leafs[2*node_thread+is_right]);
                    p1 = sample_candidate_points[accumulated_child_count[2*node_thread+is_right]  +  rand_id];
                    rand_id = (curand(&r) % count_points_on_leafs[2*node_thread+is_right]);
                    p2 = sample_candidate_points[accumulated_child_count[2*node_thread+is_right]  +  rand_id];
                    
                    while(p1 == p2 && count_points_on_leafs[2*node_thread+is_right] > K){
                        rand_id = (curand(&r) % count_points_on_leafs[2*node_thread+is_right]);
                        p2 = sample_candidate_points[accumulated_child_count[2*node_thread+is_right]  + rand_id];
                    }

                    __syncthreads();
                    create_node(node_thread, is_right, tree_new_depth, tree_parents_new_depth,
                                tree_children, tree_count, count_new_nodes, p1, p2, points, D, N);
                }
            }
        }
    }
}

#endif