#ifndef __NEAREST_NEIGHBORS_EXPLORING__CU
#define __NEAREST_NEIGHBORS_EXPLORING__CU

#include "../include/common.h"


__global__
void nearest_neighbors_exploring(typepoints* points,
                                 int* old_knn_indices,
                                 int* knn_indices,
                                 typepoints* knn_sqr_dist,
                                 int N, int D, int K)
{
    
    int p, tmp_p, tmp_candidate, max_id_point, p_neigh, p_neigh_neigh;
    int i,j,k;
    int knn_id;
    typepoints max_dist_val;

    int tid = blockDim.x*blockIdx.x+threadIdx.x;
    int lane = threadIdx.x % 32; // my id on warp
    
    #if EUCLIDEAN_DISTANCE_VERSION!=EDV_NOATOMIC_NOSHM && EUCLIDEAN_DISTANCE_VERSION!=EDV_WARP_REDUCE_XOR_NOSHM
    __shared__ typepoints candidate_dist_val[1024];
    int init_warp_on_block = threadIdx.x-lane;
    #else
    typepoints candidate_dist_val, tmp_candidate_dist_val;
    #endif



    for(p = tid; __any_sync(__activemask(), p < N); p+=blockDim.x*gridDim.x){
        if(p < N){
            knn_id = p*K;

            max_id_point = knn_id;
            max_dist_val = knn_sqr_dist[knn_id];


            for(j=1; j < K; ++j){
                if(knn_sqr_dist[knn_id+j] > max_dist_val){
                    max_id_point = knn_id+j;
                    max_dist_val = knn_sqr_dist[knn_id+j];
                }
            }
        }
        __syncwarp();
        for(i=0; i < K; ++i){
            if(p < N) p_neigh = old_knn_indices[knn_id+i];
            for(k=0; k < K; ++k){
                p_neigh_neigh = -1;

                if(p < N){
                    p_neigh_neigh = old_knn_indices[p_neigh*K + k];
                    
                    for(j=0; j < K; ++j){
                        if(p_neigh_neigh == knn_indices[knn_id+j]){
                            p_neigh_neigh = -1;
                            break;
                        }
                    }
                    #if EUCLIDEAN_DISTANCE_VERSION!=EDV_NOATOMIC_NOSHM && EUCLIDEAN_DISTANCE_VERSION!=EDV_WARP_REDUCE_XOR_NOSHM
                    candidate_dist_val[init_warp_on_block+lane] = 0.0f;
                    #else
                    candidate_dist_val = 0.0f;
                    #endif
                }

                __syncthreads();
                for(j=0; j < 32; ++j){
                    tmp_candidate = __shfl_sync(__activemask(), p_neigh_neigh, j);
                    if(tmp_candidate == -1) continue;
                    tmp_p = __shfl_sync(__activemask(), p, j);
                    #if EUCLIDEAN_DISTANCE_VERSION!=EDV_NOATOMIC_NOSHM && EUCLIDEAN_DISTANCE_VERSION!=EDV_WARP_REDUCE_XOR_NOSHM
                    euclidean_distance_sqr_coalesced(tmp_candidate, tmp_p, points, D, N,
                                                     lane, &candidate_dist_val[init_warp_on_block+j]);
                    #else
                    tmp_candidate_dist_val = euclidean_distance_sqr_coalesced(
                                                tmp_candidate, tmp_p, points, D, N,
                                                lane);
                    if(lane == j) candidate_dist_val = tmp_candidate_dist_val;
                    #endif
                }
                __syncwarp();
                if(p_neigh_neigh == -1) continue;

                #if EUCLIDEAN_DISTANCE_VERSION!=EDV_NOATOMIC_NOSHM && EUCLIDEAN_DISTANCE_VERSION!=EDV_WARP_REDUCE_XOR_NOSHM
                if(candidate_dist_val[init_warp_on_block+lane] < max_dist_val){
                #else
                if(candidate_dist_val < max_dist_val){
                #endif
                    // printf("%d %d\n", knn_indices[max_id_point] ,p_neigh_neigh);
                    knn_indices[max_id_point] = p_neigh_neigh;
                    #if EUCLIDEAN_DISTANCE_VERSION!=EDV_NOATOMIC_NOSHM && EUCLIDEAN_DISTANCE_VERSION!=EDV_WARP_REDUCE_XOR_NOSHM
                    knn_sqr_dist[max_id_point] = candidate_dist_val[init_warp_on_block+lane];
                    #else
                    knn_sqr_dist[max_id_point] = candidate_dist_val;
                    #endif

                    max_id_point = knn_id;
                    max_dist_val = knn_sqr_dist[knn_id];
                    for(j=1; j < K; ++j){
                        if(knn_sqr_dist[knn_id+j] > max_dist_val){
                            max_id_point = knn_id+j;
                            max_dist_val = knn_sqr_dist[knn_id+j];
                        }
                    }
                }
            }
        }
    }
}

#endif