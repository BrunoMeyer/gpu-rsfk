/*
This file is part of the GPU-RSFK Project (https://github.com/BrunoMeyer/gpu-rsfk).

BSD 3-Clause License

Copyright (c) 2021, Bruno Henrique Meyer, Wagner M. Nunan Zola
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef __BUILD_TREE_UPDATE_PARENTS_ANN__CU
#define __BUILD_TREE_UPDATE_PARENTS_ANN__CU

#include "../include/common.h"

__global__
void check_points_is_leaf(
    int* points_parent,
    int* points_depth,
    bool* is_leaf,
    int depth,
    int NQ)
{
    int tid = blockDim.x*blockIdx.x+threadIdx.x;
    int parent_id;
    for(int p = tid; p < NQ; p+=blockDim.x*gridDim.x){
        if(depth != points_depth[p]) continue;
        // printf("%d %d %d\n", points_depth[p], points_parent[p], accumulated_nodes_count[points_depth[p]]);
        printf("%d %d %d: %d\n", depth, p, points_parent[p], is_leaf[points_parent[p]]);
    }
}

__global__
void compute_bucket_for_query_points(
    int* points_parent,
    int* points_depth,
    int* accumulated_nodes_count,
    int* query_to_bucket_id,
    int* node_idx_to_leaf_idx,
    int NQ)
{
    int tid = blockDim.x*blockIdx.x+threadIdx.x;
    int parent_id;
    for(int p = tid; p < NQ; p+=blockDim.x*gridDim.x){
        // printf("%d %d %d\n", points_depth[p], points_parent[p], accumulated_nodes_count[points_depth[p]]);
        parent_id = accumulated_nodes_count[points_depth[p]] + points_parent[p];
        query_to_bucket_id[p] = node_idx_to_leaf_idx[parent_id];
    }
}

__global__
void
build_tree_utils_ann(int* actual_depth,
                 int* depth_level_count,
                 int* count_new_nodes,
                 int* tree_count,
                 int* accumulated_nodes_count,
                 int* device_active_points_count)
// Executed at end of each iteration of tree building
{
    *actual_depth = *actual_depth+1;
    *device_active_points_count = 0;
}

__global__
void build_tree_update_parents_ann(
    RSFK_typepoints* tree,
    int* tree_parents,
    int* tree_children,
    int* tree_children_new_depth,
    int* points_parent,
    int* points_depth,
    int* is_right_child,
    bool* is_leaf,
    bool* is_leaf_new_depth,
    int* child_count,
    int* child_count_new_depth,
    RSFK_typepoints* points,
    int* actual_depth,
    int* tree_count,
    int* depth_level_count,
    int* count_new_nodes,
    int N, int D,
    int MIN_TREE_CHILD, int MAX_TREE_CHILD,
    bool is_last_level)
{
    int tid = blockDim.x*blockIdx.x+threadIdx.x;

    int right_child, p;
    int updated_count;
    int old_parent, old_depth;
    // Set nodes parent in the new depth
    for(p = tid; p < N; p+=blockDim.x*gridDim.x){
        if(points_depth[p] == *actual_depth-1 && !is_leaf[points_parent[p]]){
            right_child = is_right_child[p];
            old_parent = points_parent[p];
            old_depth = points_depth[p];
            points_parent[p] = tree_children[2*old_parent+right_child];
            
            if(points_parent[p] == -1 && !is_leaf[points_parent[p]]){
                points_parent[p] = tree_children[2*old_parent+abs(right_child-1)];
            }

            // Possible BUG
            // if(tree_children_new_depth[points_parent[p]] == -1
            //    && tree_children_new_depth[points_parent[p]+1] == -1){
            //     points_parent[p] = tree_children[2*old_parent+abs(right_child-1)];
            // }

            if(child_count_new_depth[points_parent[p]] == 0){
                points_parent[p] = tree_children[2*old_parent+abs(right_child-1)];
                // printf("count old | tmp | new \t %d | %d | %d\n", child_count[old_parent], child_count_new_depth[tree_children[2*old_parent+right_child]], child_count_new_depth[points_parent[p]]);
            }
            // else printf("child_count_new_depth[points_parent[p]]: %d\n", child_count_new_depth[points_parent[p]]);


            points_depth[p] = *actual_depth;



            // if(is_leaf_new_depth[points_parent[p]]) printf("[%d] %d reached leaf: Parent = %d, Depth: %d\n", is_leaf_new_depth[points_parent[p]], p, points_parent[p], points_depth[p]);

            if(points_parent[p] == -1 && !is_leaf[points_parent[p]]){
                printf(">>>>> OPS, points_parent[%d] == -1. %d | %d | %d\n", p, old_parent, tree_children[2*old_parent+right_child], tree_children[2*old_parent+!right_child]);
                points_parent[p] = old_parent;
                points_depth[p] = old_depth;
            }
            // else {
            //     updated_count = atomicAdd(&child_count_new_depth[points_parent[p]],1)+1;
            // }
        }
    }
}


__device__                // NOTE: value returned in register (NO SHM)
inline                    // function return type CHANGED
RSFK_typepoints euclidean_distance_sqr_coalesced_ann(
    int pq,
    int pb,
    RSFK_typepoints* points_query,
    RSFK_typepoints* points,
    int D,
    int N,
    int lane)
{
    RSFK_typepoints diff;
    RSFK_typepoints s = 0.0f;
    
    for(int i=lane; i < D; i+=32){
        diff = points_query[get_point_idx(pq,i,N,D)] - points[get_point_idx(pb,i,N,D)];
        s+=diff*diff;
    }
    //atomicAdd(diff_sqd,s);
    s += __shfl_xor_sync( 0xffffffff, s,  1); // assuming warpSize=32
    s += __shfl_xor_sync( 0xffffffff, s,  2); // assuming warpSize=32
    s += __shfl_xor_sync( 0xffffffff, s,  4); // assuming warpSize=32
    s += __shfl_xor_sync( 0xffffffff, s,  8); // assuming warpSize=32
    s += __shfl_xor_sync( 0xffffffff, s, 16); // assuming warpSize=32
    // all lanes have the value, just return it
    return s;
}

__global__
void compute_knn_from_buckets_perwarp_coalesced_ann(
    int* points_parent,
    int* points_depth,
    int* accumulated_nodes_count,
    RSFK_typepoints* points,
    RSFK_typepoints* points_query,
    int* query_to_bucket_id,
    int* node_idx_to_leaf_idx,
    int* nodes_bucket,
    int* bucket_size,
    int* knn_indices,
    RSFK_typepoints* knn_sqr_dist,
    int N, int NQ, int D, int max_bucket_size, int K,
    int MAX_TREE_CHILD, int total_buckets)
{
    int tid = blockDim.x*blockIdx.x+threadIdx.x;
    int parent_id, current_bucket_size, max_id_point, candidate_point;
    RSFK_typepoints max_dist_val;
    
    int knn_id, tmp_knn_id;
    int lane = threadIdx.x % 32; // my id on warp
    

    #if RSFK_EUCLIDEAN_DISTANCE_VERSION!=RSFK_EDV_NOATOMIC_NOSHM && RSFK_EUCLIDEAN_DISTANCE_VERSION!=RSFK_EDV_WARP_REDUCE_XOR_NOSHM
    __shared__ RSFK_typepoints candidate_dist_val[1024];
    int init_warp_on_block = (threadIdx.x/32)*32;
    #else
    RSFK_typepoints candidate_dist_val, tmp_dist_val;
    #endif

    int bid, i, j, k;
    int pq, _pb, pb;
    int tmp_candidate, tmp_p;
    float tmp_dist;
    RSFK_typepoints tmp_max_dist_val;
    
    for(pq = tid/32; pq < NQ; pq+=blockDim.x*gridDim.x/32){
        int bid = query_to_bucket_id[pq];
        current_bucket_size = bucket_size[bid];
        knn_id = pq*K;

        // parent_id = accumulated_nodes_count[points_depth[p]] + points_parent[p];
        // current_bucket_size = bucket_size[node_idx_to_leaf_idx[parent_id]];

        max_id_point = knn_id;
        max_dist_val = knn_sqr_dist[knn_id];
        // Finds the index of the furthest point from the current result of knn_indices
        // and the distance between them
        for(j=1; j < K; ++j){
            if(knn_sqr_dist[knn_id+j] > max_dist_val){
                max_id_point = knn_id+j;
                max_dist_val = knn_sqr_dist[knn_id+j];
            }
        }
        current_bucket_size = bucket_size[bid];

        int local_max_position = -1;
        float local_max_dist = -1.0f;

        float tmp_max_dist;
        int tmp_max_position;
        
        
        for(k=lane; k < K; k+=RSFK_WarpSize){
            if(knn_sqr_dist[knn_id+k] > local_max_dist){
                local_max_position = knn_id+k;
                local_max_dist = knn_sqr_dist[knn_id+k];
            }
        }
        
        tmp_max_position = __shfl_down_sync( 0xffffffff, local_max_position,  16); // assuming warpSize=32
        tmp_max_dist = __shfl_down_sync( 0xffffffff, local_max_dist,  16); // assuming warpSize=32
        if(tmp_max_dist > local_max_dist){
            local_max_dist = tmp_max_dist;
            local_max_position = tmp_max_position;
        }
        tmp_max_dist = __shfl_down_sync( 0xffffffff, local_max_dist,  8); // assuming warpSize=32
        tmp_max_position = __shfl_down_sync( 0xffffffff, local_max_position,  8); // assuming warpSize=32
        if(tmp_max_dist > local_max_dist){
            local_max_dist = tmp_max_dist;
            local_max_position = tmp_max_position;
        }
        tmp_max_dist = __shfl_down_sync( 0xffffffff, local_max_dist,  4); // assuming warpSize=32
        tmp_max_position = __shfl_down_sync( 0xffffffff, local_max_position,  4); // assuming warpSize=32
        if(tmp_max_dist > local_max_dist){
            local_max_dist = tmp_max_dist;
            local_max_position = tmp_max_position;
        }
        tmp_max_dist = __shfl_down_sync( 0xffffffff, local_max_dist,  2); // assuming warpSize=32
        tmp_max_position = __shfl_down_sync( 0xffffffff, local_max_position,  2); // assuming warpSize=32
        if(tmp_max_dist > local_max_dist){
            local_max_dist = tmp_max_dist;
            local_max_position = tmp_max_position;
        }
        tmp_max_dist = __shfl_down_sync( 0xffffffff, local_max_dist, 1); // assuming warpSize=32
        tmp_max_position = __shfl_down_sync( 0xffffffff, local_max_position, 1); // assuming warpSize=32
        if(tmp_max_dist > local_max_dist){
            local_max_dist = tmp_max_dist;
            local_max_position = tmp_max_position;
        }

        max_id_point = __shfl_sync(0xffffffff, local_max_position, 0);
        max_dist_val = __shfl_sync(0xffffffff, local_max_dist, 0);



        for(_pb = lane; __any_sync(__activemask(), _pb < current_bucket_size); _pb+=32){
            candidate_point = -1;

            if(_pb < current_bucket_size){
                pb = nodes_bucket[bid*max_bucket_size + _pb];
                // candidate_point = nodes_bucket[node_idx_to_leaf_idx[parent_id]*max_bucket_size + i];
                candidate_point = pb;
                
                // Verify if the candidate point (inside the bucket of current point)
                // already is in the knn_indices result
                for(j=0; j < K; ++j){
                    if(candidate_point == knn_indices[knn_id+j]){
                        candidate_point = -1;
                        break;
                    }
                }
                // If it is, then it doesnt need to be treated, then go to
                // the next iteration and wait the threads from same warp to goes on
            }
            

            #if RSFK_EUCLIDEAN_DISTANCE_VERSION!=RSFK_EDV_NOATOMIC_NOSHM && RSFK_EUCLIDEAN_DISTANCE_VERSION!=RSFK_EDV_WARP_REDUCE_XOR_NOSHM
            candidate_dist_val[threadIdx.x] = 0.0f;
            #endif

            for(j=0; j < 32; ++j){
                __syncwarp();
                tmp_candidate = __shfl_sync(0xffffffff, candidate_point, j);
                if(tmp_candidate == -1) continue;
                tmp_p = __shfl_sync(0xffffffff, pq, j);
                #if RSFK_EUCLIDEAN_DISTANCE_VERSION!=RSFK_EDV_NOATOMIC_NOSHM && RSFK_EUCLIDEAN_DISTANCE_VERSION!=RSFK_EDV_WARP_REDUCE_XOR_NOSHM
                euclidean_distance_sqr_coalesced_ann(tmp_candidate, tmp_p, points, D, N,
                                                lane,
                                                &candidate_dist_val[init_warp_on_block+j]);
                #else
                tmp_dist_val = euclidean_distance_sqr_coalesced_ann(tmp_p, tmp_candidate, points, points_query, D, N, lane);
                if(lane == j) candidate_dist_val = tmp_dist_val;
                #endif
            }
            // if(candidate_point == -1) continue;

            // If the candidate is closer than the pre-computed furthest point,
            // switch them
            // #if RSFK_EUCLIDEAN_DISTANCE_VERSION!=RSFK_EDV_NOATOMIC_NOSHM && RSFK_EUCLIDEAN_DISTANCE_VERSION!=RSFK_EDV_WARP_REDUCE_XOR_NOSHM
            // if(candidate_dist_val[threadIdx.x] < max_dist_val){
            // #else
            // if(candidate_dist_val < max_dist_val){
            // #endif
                // knn_indices[max_id_point] = candidate_point;
                // #if RSFK_EUCLIDEAN_DISTANCE_VERSION!=RSFK_EDV_NOATOMIC_NOSHM && RSFK_EUCLIDEAN_DISTANCE_VERSION!=RSFK_EDV_WARP_REDUCE_XOR_NOSHM
                // knn_sqr_dist[max_id_point] = candidate_dist_val[threadIdx.x];
                // #else
                // knn_sqr_dist[max_id_point] = candidate_dist_val;
                // #endif
                // Also update the furthest point that will be used in the next
                // comparison
                
                /*
                max_id_point = knn_id;
                max_dist_val = knn_sqr_dist[knn_id];
                for(j=1; j < K; ++j){
                    if(knn_sqr_dist[knn_id+j] > max_dist_val){
                        max_id_point = knn_id+j;
                        max_dist_val = knn_sqr_dist[knn_id+j];
                    }
                }
                printf("%d %d %f %f %f\n", pq, max_id_point-knn_id, max_dist_val, knn_sqr_dist[knn_id+3], candidate_dist_val);
                */
            // }
            for(j=0; j < 32; ++j){
                __syncwarp();
                tmp_candidate = __shfl_sync(0xffffffff, candidate_point, j);
                if(tmp_candidate == -1) continue;
                tmp_dist = __shfl_sync(0xffffffff, candidate_dist_val, j);
                tmp_max_dist_val = __shfl_sync(0xffffffff, max_dist_val, j);

                if(tmp_dist < tmp_max_dist_val){
                    if(lane == j){
                        knn_indices[max_id_point] = candidate_point;
                        knn_sqr_dist[max_id_point] = candidate_dist_val;
                    }
                    // if(lane == 0){
                    //     knn_indices[max_id_point] = tmp_candidate;
                    //     knn_sqr_dist[max_id_point] = tmp_dist;
                    // }

                    __syncwarp();
                    int local_max_position = -1;
                    float local_max_dist = -1.0f;

                    float tmp_max_dist;
                    int tmp_max_position;
                    
                    // if(lane == 0){
                    //     for(k=0; k < K; k+=1){
                    //         if(knn_sqr_dist[tmp_knn_id+k] > local_max_dist){
                    //             local_max_position = tmp_knn_id+k;
                    //             local_max_dist = knn_sqr_dist[tmp_knn_id+k];
                    //         }
                    //     }
                    // }
                    
                    for(k=lane; k < K; k+=RSFK_WarpSize){
                        if(knn_sqr_dist[knn_id+k] > local_max_dist){
                            local_max_position = knn_id+k;
                            local_max_dist = knn_sqr_dist[knn_id+k];
                        }
                    }
                    
                    tmp_max_position = __shfl_down_sync( 0xffffffff, local_max_position,  16); // assuming warpSize=32
                    tmp_max_dist = __shfl_down_sync( 0xffffffff, local_max_dist,  16); // assuming warpSize=32
                    if(tmp_max_dist > local_max_dist){
                        local_max_dist = tmp_max_dist;
                        local_max_position = tmp_max_position;
                    }
                    tmp_max_dist = __shfl_down_sync( 0xffffffff, local_max_dist,  8); // assuming warpSize=32
                    tmp_max_position = __shfl_down_sync( 0xffffffff, local_max_position,  8); // assuming warpSize=32
                    if(tmp_max_dist > local_max_dist){
                        local_max_dist = tmp_max_dist;
                        local_max_position = tmp_max_position;
                    }
                    tmp_max_dist = __shfl_down_sync( 0xffffffff, local_max_dist,  4); // assuming warpSize=32
                    tmp_max_position = __shfl_down_sync( 0xffffffff, local_max_position,  4); // assuming warpSize=32
                    if(tmp_max_dist > local_max_dist){
                        local_max_dist = tmp_max_dist;
                        local_max_position = tmp_max_position;
                    }
                    tmp_max_dist = __shfl_down_sync( 0xffffffff, local_max_dist,  2); // assuming warpSize=32
                    tmp_max_position = __shfl_down_sync( 0xffffffff, local_max_position,  2); // assuming warpSize=32
                    if(tmp_max_dist > local_max_dist){
                        local_max_dist = tmp_max_dist;
                        local_max_position = tmp_max_position;
                    }
                    tmp_max_dist = __shfl_down_sync( 0xffffffff, local_max_dist, 1); // assuming warpSize=32
                    tmp_max_position = __shfl_down_sync( 0xffffffff, local_max_position, 1); // assuming warpSize=32
                    if(tmp_max_dist > local_max_dist){
                        local_max_dist = tmp_max_dist;
                        local_max_position = tmp_max_position;
                    }
                    // if(lane == 0 || lane == j){
                    //     __shfl_down_sync( 0xffffffff, s,  1 );
                    // }

                    // if(lane == 0){
                    //     printf("%d %d %d %d %f %f \n", tmp_p, current_bucket_size, max_id_point, local_max_position, max_dist_val, local_max_dist);
                    // }
                    max_id_point = __shfl_sync(0xffffffff, local_max_position, 0);
                    max_dist_val = __shfl_sync(0xffffffff, local_max_dist, 0);
                    // if(lane == j){
                        // printf("%d %d %d %d %f %f \n", pq, current_bucket_size, max_id_point, local_max_position, max_dist_val, local_max_dist);
                        // max_id_point = tmp_max_position;
                        // max_dist_val = tmp_max_dist;
                        // printf("%d %d %f %f %f\n", pq, max_id_point-tmp_knn_id, max_dist_val, knn_sqr_dist[knn_id+3], candidate_dist_val);
                    // }
                }
            }
            
            
        }
    }
}

#endif