#ifndef __BUILD_TREE_CHECK_POINTS_SIDE__CU
#define __BUILD_TREE_CHECK_POINTS_SIDE__CU

#include "../include/common.h"

__device__
inline
int check_hyperplane_side(int node_idx,
                          int p,
                          typepoints* tree,
                          typepoints* points,
                          int D,
                          int N,
                          int* count_new_nodes)
{
    typepoints aux = 0.0f;
    for(int i=0; i < D; ++i){
        // aux += tree[node_idx*(D+1) + i]*points[p*D + i];
        aux += tree[get_tree_idx(node_idx,i,*count_new_nodes,D)]*points[get_point_idx(p,i,N,D)];
    }
    return aux < tree[get_tree_idx(node_idx,D,*count_new_nodes,D)];
}

// #if EUCLIDEAN_DISTANCE_VERSION==EDV_WARP_REDUCE_XOR
__device__
inline
void check_hyperplane_side_coalesced(int node_idx, int p, typepoints* tree,
                                    typepoints* points, int D, int N,
                                    int* count_new_nodes,
                                    int tidw, typepoints* product)
{
    typepoints s = 0.0f;
    for(int i=tidw; i < D; i+=32){
        s+=tree[get_tree_idx(node_idx,i,*count_new_nodes,D)]*points[get_point_idx(p,i,N,D)];
    }

    s += __shfl_xor_sync( 0xffffffff, s,  1); // assuming warpSize=32
    s += __shfl_xor_sync( 0xffffffff, s,  2); // assuming warpSize=32
    s += __shfl_xor_sync( 0xffffffff, s,  4); // assuming warpSize=32
    s += __shfl_xor_sync( 0xffffffff, s,  8); // assuming warpSize=32
    s += __shfl_xor_sync( 0xffffffff, s, 16); // assuming warpSize=32
    // lane 0 stores result in SHM
    if( threadIdx.x & 0x1f )
        *product = s;
}

// __device__
// inline
// typepoints check_hyperplane_side_coalesced(int node_idx, int p, typepoints* tree,
//                                     typepoints* points, int D, int N,
//                                     int* count_new_nodes,
//                                     int tidw)
// {
//     typepoints s = 0.0f;
//     for(int i=tidw; i < D; i+=32){
//         s+=tree[get_tree_idx(node_idx,i,*count_new_nodes,D)]*points[get_point_idx(p,i,N,D)];
//     }

//     s += __shfl_down_sync( 0xffffffff, s, 16 ); // assuming warpSize=32
//     s += __shfl_down_sync( 0xffffffff, s,  8 ); // assuming warpSize=32
//     s += __shfl_down_sync( 0xffffffff, s,  4 ); // assuming warpSize=32
//     s += __shfl_down_sync( 0xffffffff, s,  2 ); // assuming warpSize=32
//     s += __shfl_down_sync( 0xffffffff, s,  1 ); // assuming warpSize=32
//     // broadcast reduced value to all threads in warp (so they can return the value)
//     return s;
// }
// #elif 
// #endif

__global__
void build_tree_check_active_points(int* points_parent,
                                    int* points_depth,
                                    bool* is_leaf,
                                    int* actual_depth,
                                    int* active_points,
                                    int* active_points_count,
                                    int N)
{
    int p;
    int pa_id;
    int tid = blockDim.x*blockIdx.x+threadIdx.x;
    for(p = tid; p < N; p+=blockDim.x*gridDim.x){
        if(points_depth[p] >= *actual_depth-1 && !is_leaf[points_parent[p]]){
            pa_id = atomicAdd(active_points_count, 1);
            active_points[pa_id] = p;
        }
    }
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
                                  int* active_points,
                                  int* active_points_count,
                                  int* count_new_nodes,
                                  int N, int D, int RANDOM_SEED)
{
    int tid = blockDim.x*blockIdx.x+threadIdx.x;

    int i,p, is_right;
    int csi; //candidate_sample_id;
    
    // Set nodes parent in the new depth
    for(i = tid; i < *active_points_count; i+=blockDim.x*gridDim.x){
        // if(points_depth[p] < *actual_depth-1 || is_leaf[points_parent[p]]) continue;
        p = active_points[i];
        // __syncthreads();
        
        is_right = check_hyperplane_side(points_parent[p], p, tree, points, D, N, count_new_nodes);
        
        is_right_child[p] = is_right;
        
        csi = atomicAdd(&count_points_on_leafs[2*points_parent[p]+is_right], 1);
        points_id_on_sample[p] = csi;
    }
}

__global__
void build_tree_check_points_side_coalesced(typepoints* tree,
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
                                  int* active_points,
                                  int* active_points_count,
                                  int* count_new_nodes,
                                  int N, int D, int RANDOM_SEED)
{
    int tid = blockDim.x*blockIdx.x+threadIdx.x;

    int i, j, p, tmp_p;
    int csi; //candidate_sample_id;

    __shared__ typepoints product_threads[1024];
    // typepoints product, tmp_product; 

    int tidw = threadIdx.x % 32; // my id on warp
    int init_warp_on_block = threadIdx.x-tidw;

    // Set nodes parent in the new depth
    for(i = tid; __any_sync(__activemask(), i < *active_points_count); i+=blockDim.x*gridDim.x){
        // if(points_depth[p] < *actual_depth-1 || is_leaf[points_parent[p]]) continue;
        p = -1;
        if(i < *active_points_count) p = active_points[i];
        

        // __syncthreads();
        // __syncwarp();
        product_threads[init_warp_on_block + tidw] = 0.0f;
        for(j=0; j < 32; ++j){
            tmp_p = __shfl_sync(__activemask(), p, j);
            if(tmp_p == -1) continue;
            __syncthreads();
            check_hyperplane_side_coalesced(points_parent[tmp_p], tmp_p, tree, points, D, N,
                                            count_new_nodes,
                                            tidw,
                                            &product_threads[init_warp_on_block + j]);
            
            // tmp_product = check_hyperplane_side_coalesced(points_parent[tmp_p], tmp_p, tree, points, D, N,
            //                                               count_new_nodes,
            //                                               tidw);
                
            // if(j == wid) product = tmp_product;
        }
        __syncwarp();
        if(p == -1) continue;
        
        // is_right_child[p] = product_threads[threadIdx.x] < tree[points_parent[p]*(D+1) + D];
        is_right_child[p] = product_threads[init_warp_on_block + tidw] < tree[get_tree_idx(points_parent[p],D,*count_new_nodes,D)];
        // is_right_child[p] = product < tree[get_tree_idx(points_parent[p],D,*count_new_nodes,D)];
        
        csi = atomicAdd(&count_points_on_leafs[2*points_parent[p]+is_right_child[p]], 1);
        points_id_on_sample[p] = csi;
    }
}

#endif