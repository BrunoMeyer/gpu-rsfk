#ifndef __BUILD_TREE_CHECK_POINTS_SIDE__CU
#define __BUILD_TREE_CHECK_POINTS_SIDE__CU


__device__
inline
int check_hyperplane_side(int node_idx, int p, typepoints* tree, typepoints* points, int D)
{
    typepoints aux = 0.0f;
    for(int i=0; i < D; ++i){
        aux += tree[node_idx*(D+1) + i]*points[p*D + i];
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
                                  int* sample_points,
                                  int* child_count,
                                  typepoints* points,
                                  int* actual_depth,
                                  int* tree_count,
                                  int* depth_level_count,
                                  int N, int D)
{
    // curandState_t r; 
    // curand_init(RANDOM_SEED+tid, // the seed controls the sequence of random values that are produced
    //         blockIdx.x,  // the sequence number is only important with multiple cores 
    //         tid,  // the offset is how much extra we advance in the sequence for each call, can be 0 
    //         &r);


    int tid = blockDim.x*blockIdx.x+threadIdx.x;
    int p, is_right;

    // Set nodes parent in the new depth
    for(p = tid; p < N; p+=blockDim.x*gridDim.x){
        // __syncwarp();
        // __syncthreads();
        
        if(points_depth[p] < *actual_depth-1) continue;
        
        is_right = check_hyperplane_side(points_parent[p], p, tree, points, D);
        is_right_child[p] = is_right;

        // Threats to Validity: This assumes that all the follow properties are false:
        // - The atomic operations assumes an arbitrary/random order
        // - The points are shuffled

        if(sample_points[4*points_parent[p]  + 2*is_right    ] == -1){
            sample_points[4*points_parent[p]  + 2*is_right    ] = N;
        }

        // device_sample_points[4*points_parent[p]  + 2*is_right + curand(&r) % 2] = p;

        atomicMin(&sample_points[4*points_parent[p]  + 2*is_right    ], p);
        atomicMax(&sample_points[4*points_parent[p]  + 2*is_right + 1], p);
    }
}

#endif