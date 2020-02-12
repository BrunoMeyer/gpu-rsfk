#ifndef __BUILD_TREE_CREATE_NODES__CU
#define __BUILD_TREE_CREATE_NODES__CU

#include "create_node.cu"

__global__
void build_tree_create_nodes(typepoints* tree_new_depth,
                             int* tree_parents_new_depth,
                             int* tree_children,
                             int* points_parent,
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
                             int N, int D)
{
    int tid = blockDim.x*blockIdx.x+threadIdx.x;
    // curandState_t r; 
    // curand_init(RANDOM_SEED+tid, // the seed controls the sequence of random values that are produced
    //         blockIdx.x,  // the sequence number is only important with multiple cores 
    //         tid,  // the offset is how much extra we advance in the sequence for each call, can be 0 
    //         //   &states[blockIdx.x]);
    //         &r);

    int p1, p2;
    int i, node_thread;

    // if(tid == 0){
    //     printf("DEBUG 7: %d %d %d %d\n", depth_level_count[*actual_depth-1], depth_level_count[*actual_depth], *count_new_nodes, *actual_depth);
    // }
    // Create new nodes
    for(node_thread = tid; node_thread < depth_level_count[*actual_depth-1]; node_thread+=blockDim.x*gridDim.x){

        // int parent_id = tree_parents[node_thread];
        // if(is_leaf[parent_id] && device_child_count[parent_id] > MAX_TREE_CHILD){
        
        // printf("DEBUG3: %d %d\n", node_thread, child_count[node_thread]);
        if(child_count[node_thread] > MAX_TREE_CHILD){
            for(int is_right=0; is_right < 2; ++is_right){
                p1 = sample_points[4*node_thread + 2*is_right];
                // p1 = curand(&r) % N;
                // int init_p_search = p1 != -1 ? p1 : 0;
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
                // init_p_search = p2 != -1 ? p2 : 0;
                // while((p1 == p2  || points_parent[p2] != node_thread)){
                //     p2=((p2+1)%N);
                //     if(p2 == init_p_search){
                //         p2 = -1;
                //         break;
                //     }
                // }
                if(p1 != -1 && p2 != -1){
                    // create_node(node_thread, p1, p2, tree, points, D);
                    create_node(node_thread, is_right, tree_new_depth, tree_parents_new_depth,
                                tree_children, tree_count, count_new_nodes, p1, p2, points, D);

                    is_leaf[node_thread] = false;
                    is_leaf_new_depth[tree_children[2*node_thread+is_right]] = true;

                    // is_leaf[node_thread] = true;
                    // if(child_count[parent_id] > 2500) printf("%d\n", child_count[parent_id]);
                }
                else{
                    // if(device_child_count[parent_id] > 200)
                    printf("%d %d %d | %d %d\n",*actual_depth, node_thread, child_count[node_thread], p1,p2);
                    child_count[node_thread] = 0;
                    // printf("BBBBBBBBBBBBB\n");
                }
            }
        }
    }
}

#endif