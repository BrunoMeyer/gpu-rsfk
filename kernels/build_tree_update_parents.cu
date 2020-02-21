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
                               bool* is_leaf_new_depth,
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

    int right_child, p;
    int updated_count;
    // Set nodes parent in the new depth
    for(p = tid; p < N; p+=blockDim.x*gridDim.x){
        // TODO: Verify if the "if" statement is necessary
        // if(points_depth[p] == *actual_depth-1 && child_count[points_parent[p]] > 2*MAX_TREE_CHILD){
        if(points_depth[p] == *actual_depth-1 && !is_leaf[points_parent[p]]){
            right_child = is_right_child[p];
            // if(tree_children[2*points_parent[p]+right_child] == -1) printf("%s: line %d: %d\n",__FILE__,__LINE__, points_parent[p]);
            // if(*actual_depth > 2000){
            //     if(right_child)printf("%s: line %d: %d %d %d\n",__FILE__,__LINE__, p, child_count[points_parent[p]], right_child);
            // }
            points_parent[p] = tree_children[2*points_parent[p]+right_child];
            updated_count = atomicAdd(&child_count_new_depth[points_parent[p]],1)+1;
            points_depth[p] = *actual_depth;

            if(updated_count <= 2*MAX_TREE_CHILD){
                is_leaf_new_depth[points_parent[p]] = true;
            }
            else{
                is_leaf_new_depth[points_parent[p]] = false;
            }

            
            
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
                                    bool* is_leaf_new_depth,
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
    int parent_leaf_node;
    int new_count;
    // Set nodes parent in the new depth
    for(p = tid; p < N; p+=blockDim.x*gridDim.x){
        // if(points_depth[p] == *actual_depth && child_count_new_depth[points_parent[p]] <= 2*MAX_TREE_CHILD){
        //     // is_leaf[points_parent[p]] = true;
        //     // right_child = is_right_child[p];

        //     // parent_leaf_node = 2*tree_parents_new_depth[points_parent[p]];
        //     // is_leaf_new_depth[tree_children[parent_leaf_node+0]] = true;
        //     // is_leaf_new_depth[tree_children[parent_leaf_node+1]] = true;
        //     is_leaf_new_depth[points_parent[p]] = true;
        // }
        if(points_depth[p] == *actual_depth && child_count_new_depth[points_parent[p]] <= K){
            child_count_new_depth[points_parent[p]] = 0;
            is_leaf_new_depth[points_parent[p]] = false;
            // is_leaf_new_depth[points_parent[p]] = false;

            right_child = abs(is_right_child[p]-1);
            parent_leaf_node = 2*tree_parents_new_depth[points_parent[p]];
            
            // if(tree_children[parent_leaf_node+right_child] == -1) printf("\n\n%s: line %d: %d\n\n",__FILE__,__LINE__, points_parent[p]);
            points_parent[p] = tree_children[parent_leaf_node+right_child];

            is_leaf_new_depth[points_parent[p]] = true;
            new_count = atomicAdd(&child_count_new_depth[points_parent[p]],1)+1;
            // if(new_count > 150) printf("%s: line %d: %d\n",__FILE__,__LINE__, new_count);
            if(new_count > 2*MAX_TREE_CHILD){
                // if(*actual_depth > 2000){
                //     printf("%s: line %d: %d %d\n",__FILE__,__LINE__, p, new_count);
                // }
                is_leaf_new_depth[points_parent[p]] = false;
            }

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