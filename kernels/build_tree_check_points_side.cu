#ifndef __BUILD_TREE_CHECK_POINTS_SIDE__CU
#define __BUILD_TREE_CHECK_POINTS_SIDE__CU


__device__
inline
int check_hyperplane_side(int node_idx, int p, typepoints* tree, typepoints* points, int D, int N)
{
    typepoints aux = 0.0f;
    for(int i=0; i < D; ++i){
        // aux += tree[node_idx*(D+1) + i]*points[p*D + i];
        aux += tree[node_idx*(D+1) + i]*points[get_point_idx(p,i,N,D)];
    }
    return aux < tree[node_idx*(D+1) + D];
}


__global__
void build_tree_check_points_side(typepoints* tree,
                                  int* tree_parents,
                                  int* tree_children,
                                  int* points_parent,
                                  int* points_depth,
                                  int* is_right_child,
                                  bool* is_leaf,
                                  int* child_count,
                                  typepoints* points,
                                  int* actual_depth,
                                  int* tree_count,
                                  int* depth_level_count,
                                  int* accumulated_child_count,
                                  int* count_points_on_leafs,
                                  int* sample_candidate_points,
                                  int* points_id_on_sample,
                                  int N, int D, int RANDOM_SEED)
{
    int tid = blockDim.x*blockIdx.x+threadIdx.x;
    curandState_t r; 
    curand_init(*actual_depth*(RANDOM_SEED+blockDim.x)+RANDOM_SEED+tid, // the seed controls the sequence of random values that are produced
            blockIdx.x,  // the sequence number is only important with multiple cores 
            tid,  // the offset is how much extra we advance in the sequence for each call, can be 0 
            &r);


    int p, is_right;
    int csi; //candidate_sample_id;
    // Set nodes parent in the new depth
    for(p = tid; p < N; p+=blockDim.x*gridDim.x){
        if(points_depth[p] < *actual_depth-1 || is_leaf[points_parent[p]]) continue;
        
        __syncthreads();
        // __syncwarp();
        is_right = check_hyperplane_side(points_parent[p], p, tree, points, D, N);
        is_right_child[p] = is_right;
        
        csi = atomicAdd(&count_points_on_leafs[2*points_parent[p]+is_right], 1);
        points_id_on_sample[p] = csi;
        
    }
}

#endif