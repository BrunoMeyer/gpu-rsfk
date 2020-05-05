#ifndef __BUILD_TREE_COUNT_NEW_NODES__CU
#define __BUILD_TREE_COUNT_NEW_NODES__CU

__global__
void build_tree_count_new_nodes(typepoints* tree,
                                int* tree_parents,
                                int* tree_children,
                                int* points_parent,
                                int* is_right_child,
                                bool* is_leaf,
                                int* count_points_on_leafs,
                                int* child_count,
                                typepoints* points,
                                int* actual_depth,
                                int* tree_count,
                                int* depth_level_count,
                                int* count_new_nodes,
                                int N, int D,
                                int MIN_TREE_CHILD, int MAX_TREE_CHILD)
{
    int tid = blockDim.x*blockIdx.x+threadIdx.x;
    // printf("%s: line %d : %d %d \n", __FILE__, __LINE__, depth_level_count[*actual_depth-1], *actual_depth);
    for(int node_thread = tid; node_thread < depth_level_count[*actual_depth-1]; node_thread+=blockDim.x*gridDim.x){
        if(!is_leaf[node_thread] && child_count[node_thread] > 0){
            // if((sample_points[4*node_thread+0] != -1 && sample_points[4*node_thread+1] != -1)) atomicAdd(count_new_nodes, 1);
            // if((sample_points[4*node_thread+2] != -1 && sample_points[4*node_thread+3] != -1)) atomicAdd(count_new_nodes, 1);
            // if((sample_points[4*node_thread+0] != -1 && sample_points[4*node_thread+1] != -1) ||
            //    (sample_points[4*node_thread+2] != -1 && sample_points[4*node_thread+3] != -1)) atomicAdd(count_new_nodes, 2);
            if(count_points_on_leafs[2*node_thread] > 0 || count_points_on_leafs[2*node_thread+1] > 0) atomicAdd(count_new_nodes, 2);
            
            // printf("%s: line %d : %d %d %d\n", __FILE__, __LINE__, node_thread, is_leaf[node_thread], depth_level_count[*actual_depth-1]);
        }
    }
}

__global__
void build_tree_accumulate_child_count(int* depth_level_count,
                                       int* count_new_nodes,
                                       int* count_points_on_leafs,
                                       int* accumulated_child_count,
                                       int* actual_depth)
{
    accumulated_child_count[0] = 0;
    for(int i=1; i < 2*depth_level_count[*actual_depth-1]; ++i){
        accumulated_child_count[i] = accumulated_child_count[i-1] + count_points_on_leafs[i-1];
    }
}

__global__
void build_tree_organize_sample_candidates(int* points_parent,
                                           int* points_depth,
                                           bool* is_leaf,
                                           int* is_right_child,
                                           int* count_new_nodes,
                                           int* count_points_on_leafs,
                                           int* accumulated_child_count,
                                           int* sample_candidate_points,
                                           int* points_id_on_sample,
                                           int* actual_depth,
                                           int N)
{
    

    int tid = blockDim.x*blockIdx.x+threadIdx.x;

    int p, is_right;
    int csi; //candidate_sample_id;
    // Set nodes parent in the new depth
    
    for(p = tid; p < N; p+=blockDim.x*gridDim.x){
        if(points_depth[p] < *actual_depth-1 || is_leaf[points_parent[p]]) continue;
        csi = points_id_on_sample[p];
        is_right = is_right_child[p];
        sample_candidate_points[accumulated_child_count[2*points_parent[p]+is_right]+csi] = p;
    }
}

#endif