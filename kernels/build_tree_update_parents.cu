#ifndef __BUILD_TREE_UPDATE_PARENTS__CU
#define __BUILD_TREE_UPDATE_PARENTS__CU


__global__
void build_tree_update_parents(typepoints* tree,
                               int* tree_parents,
                               int* tree_children,
                               int* points_parent,
                               int* points_depth,
                               int* is_right_child,
                               bool* is_leaf,
                               int* sample_points,
                               int* child_count,
                               int* child_count_new_depth,
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

    int right_child, p;

    // Set nodes parent in the new depth
    for(p = tid; p < N; p+=blockDim.x*gridDim.x){
        // Heap left and right nodes are separeted by 1
        // printf(">%d %d\n", points_parent[p], is_leaf[points_parent[p]]);
        // if(is_leaf[points_parent[p]]){

        // TODO: Verify if the "if" statement is necessary
        if(points_depth[p] == *actual_depth-1 && child_count[points_parent[p]] > MAX_TREE_CHILD){
            right_child = is_right_child[p];
            // printf("DEBUG4: %d %d %d %d\n", points_parent[p], 2*points_parent[p]+right_child, tree_children[2*points_parent[p]+right_child], child_count[points_parent[p]]);
            // points_parent[p] = HEAP_LEFT(points_parent[p])+right_child;
            points_parent[p] = tree_children[2*points_parent[p]+right_child];
            // printf("%d %d %d\n", p, is_right_child[p], points_parent[p]);
            atomicAdd(&child_count_new_depth[points_parent[p]],1);
            points_depth[p] = *actual_depth;
        }
        // __syncwarp();
        // __syncthreads();
    }
}

__global__
void build_tree_post_update_parents(typepoints* tree,
                                    int* tree_parents,
                                    int* tree_children,
                                    int* points_parent,
                                    int* points_depth,
                                    int* is_right_child,
                                    bool* is_leaf,
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

    int p;

    
    // Set nodes parent in the new depth
    for(p = tid; p < N; p+=blockDim.x*gridDim.x){
        // device_sample_points[2*points_parent[p]] = p;
        // device_sample_points[2*points_parent[p]+1] = p;

        if(points_depth[p] == *actual_depth-1 && child_count[points_parent[p]] <= MAX_TREE_CHILD){
            is_leaf[points_parent[p]] = true;
            // printf("DEBUG5: %d\n", points_parent[p]);
        }

        // Heap left and right nodes are separeted by 1
        // if(device_sample_points[2*points_parent[p]]     == -1 &&
        //    device_sample_points[2*points_parent[p] + 1] != p){
        //         device_sample_points[2*points_parent[p]] = p;
        // }
        // if(device_sample_points[2*points_parent[p] + 1] == -1 &&
        //    device_sample_points[2*points_parent[p]]     != p){
        //         device_sample_points[2*points_parent[p]+1] = p;
        // }
    }
}

#endif