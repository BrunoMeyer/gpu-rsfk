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
                             int* sample_points,
                             int* child_count,
                             typepoints* points,
                             int* actual_depth,
                             int* tree_count,
                             int* depth_level_count,
                             int* count_new_nodes,
                             int* accumulated_nodes_count,
                             int N, int D, int MAX_TREE_CHILD, int RANDOM_SEED)
{
    int tid = blockDim.x*blockIdx.x+threadIdx.x;
    curandState_t r; 
    curand_init(*actual_depth*(RANDOM_SEED+blockDim.x)+RANDOM_SEED+tid, // the seed controls the sequence of random values that are produced
            blockIdx.x,  // the sequence number is only important with multiple cores 
            tid,  // the offset is how much extra we advance in the sequence for each call, can be 0 
            &r);

    int p1, p2, node_thread;

    // Create new nodes
    for(node_thread = tid; node_thread < depth_level_count[*actual_depth-1]; node_thread+=blockDim.x*gridDim.x){

        if(child_count[node_thread] > MAX_TREE_CHILD){
            for(int is_right=0; is_right < 2; ++is_right){
                p1 = sample_points[4*node_thread + 2*is_right];
                // p1 = curand(&r) % N;
                // int init_p_search = p1;
                // while(points_parent[p1] != node_thread){
                //     p1=((p1+1)%N);
                //     if(p1 == init_p_search){
                //         p1 = -1;
                //         break;
                //     }
                // }

                p2 = sample_points[4*node_thread + 2*is_right + 1];
                // p2 = curand(&r) % N;
                // p2 = p1+1 % N;
                // if(p1 == p2) p2=(p2+1)%N;

                // Ensure that two different points was sampled
                // // init_p_search = p2 != -1 ? p2 : 0;
                // init_p_search = p2;
                // while(p1 != -1 && (p1 == p2  || points_parent[p2] != node_thread)){
                //     p2=((p2+1)%N);
                //     if(p2 == init_p_search){
                //         p2 = -1;
                //         break;
                //     }
                // }
                // printf("%s: line %d: %d %d\n", __FILE__, __LINE__, p1, p2);

                if(p1 != -1 && p2 != -1){
                    create_node(node_thread, is_right, tree_new_depth, tree_parents_new_depth,
                                tree_children, tree_count, count_new_nodes, p1, p2, points, D);

                    is_leaf[node_thread] = false;
                    // is_leaf_new_depth[tree_children[2*node_thread+is_right]] = true;
                }
                else{
                    printf("%s: line %d: %d %d\n", __FILE__, __LINE__, p1, p2);
                }
            }
        }
    }
}

#endif