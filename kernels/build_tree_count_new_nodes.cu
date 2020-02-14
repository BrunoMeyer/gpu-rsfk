#ifndef __BUILD_TREE_COUNT_NEW_NODES__CU
#define __BUILD_TREE_COUNT_NEW_NODES__CU

__global__
void build_tree_count_new_nodes(typepoints* tree,
                                int* tree_parents,
                                int* tree_children,
                                int* points_parent,
                                int* is_right_child,
                                bool* is_leaf,
                                int* sample_points,
                                int* child_count,
                                typepoints* points,
                                int* actual_depth,
                                int* tree_count,
                                int* depth_level_count,
                                int* count_new_nodes,
                                int N, int D, int MAX_TREE_CHILD)
{
    int tid = blockDim.x*blockIdx.x+threadIdx.x;
    for(int node_thread = tid; node_thread < depth_level_count[*actual_depth-1]; node_thread+=blockDim.x*gridDim.x){
        if(child_count[node_thread] > MAX_TREE_CHILD){
            if((sample_points[4*node_thread+0] != -1 && sample_points[4*node_thread+1] != -1)) atomicAdd(count_new_nodes, 1);
            if((sample_points[4*node_thread+2] != -1 && sample_points[4*node_thread+3] != -1)) atomicAdd(count_new_nodes, 1);
        }
    }
}

#endif