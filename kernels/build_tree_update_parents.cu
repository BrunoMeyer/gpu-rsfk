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
                               int N, int D, int MAX_TREE_CHILD)
{
    int tid = blockDim.x*blockIdx.x+threadIdx.x;

    int right_child, p;

    // Set nodes parent in the new depth
    for(p = tid; p < N; p+=blockDim.x*gridDim.x){

        // TODO: Verify if the "if" statement is necessary
        if(points_depth[p] == *actual_depth-1 && child_count[points_parent[p]] > MAX_TREE_CHILD){
            right_child = is_right_child[p];
            // printf("%s: line %d: %d\n",__FILE__,__LINE__, tree_children[2*points_parent[p]+right_child]);
            points_parent[p] = tree_children[2*points_parent[p]+right_child];
            atomicAdd(&child_count_new_depth[points_parent[p]],1);
            points_depth[p] = *actual_depth;
        }
        // __syncwarp();
        // __syncthreads();
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
                                    int* sample_points,
                                    int* child_count,
                                    int* child_count_new_depth,
                                    typepoints* points,
                                    int* actual_depth,
                                    int* tree_count,
                                    int* depth_level_count,
                                    int* count_new_nodes,
                                    int N, int D, int K, int MAX_TREE_CHILD)
{
    int tid = blockDim.x*blockIdx.x+threadIdx.x;

    int p, right_child/*, new_path*/;

    
    // Set nodes parent in the new depth
    for(p = tid; p < N; p+=blockDim.x*gridDim.x){
        if(points_depth[p] == *actual_depth-1 && child_count[points_parent[p]] <= MAX_TREE_CHILD){
            is_leaf[points_parent[p]] = true;
        }
        if(points_depth[p] == *actual_depth && child_count_new_depth[points_parent[p]] < K+1){
            right_child = abs(is_right_child[p]-1);
            child_count_new_depth[points_parent[p]] = 0;

            points_parent[p] = tree_children[2*tree_parents_new_depth[points_parent[p]]+right_child];
            atomicAdd(&child_count_new_depth[points_parent[p]],1);
            /*
            new_path = abs(is_right_child[p]-1);
            is_right_child[p] = new_path;
            child_count_new_depth[points_parent[p]] = 0;
            points_parent[p] = tree_children[2*tree_parents_new_depth[points_parent[p]]+new_path];
            is_leaf[points_parent[p]] = true;
            // printf("points_parent[p] %d\n", points_parent[p]);
            atomicAdd(&child_count_new_depth[points_parent[p]],1);
            */
        }
    }
}

#endif