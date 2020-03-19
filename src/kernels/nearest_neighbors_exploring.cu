#ifndef __NEAREST_NEIGHBORS_EXPLORING__CU
#define __NEAREST_NEIGHBORS_EXPLORING__CU


__global__
void nearest_neighbors_exploring(typepoints* points,
                                 int* old_knn_indices,
                                 int* knn_indices,
                                 typepoints* knn_sqr_dist,
                                 int N, int D, int K)
{
    int tid = blockDim.x*blockIdx.x+threadIdx.x;

    int max_id_point, p_neigh, p_neigh_neigh;
    typepoints max_dist_val, tmp_dist_val;
    
    int knn_id;

    for(int p = tid; p < N; p+=blockDim.x*gridDim.x){
        knn_id = p*K;
        // __syncthreads();
        // __syncwarp();

        max_id_point = knn_id;
        max_dist_val = knn_sqr_dist[knn_id];


        for(int j=1; j < K; ++j){
            if(knn_sqr_dist[knn_id+j] > max_dist_val){
                max_id_point = knn_id+j;
                max_dist_val = knn_sqr_dist[knn_id+j];
            }
        }
        
        for(int i=0; i < K; ++i){
            __syncthreads();
            p_neigh = old_knn_indices[knn_id+i];
            for(int k=0; k < K; ++k){
                p_neigh_neigh = old_knn_indices[p_neigh*K + k];
                if(p == p_neigh_neigh) continue;
                
                for(int j=0; j < K; ++j){
                    if(p_neigh_neigh == knn_indices[knn_id+j]){
                        p_neigh_neigh = -1;
                        break;
                    }
                }
                if(p_neigh_neigh == -1) continue;

                // tmp_dist_val = euclidean_distance_sqr(&points[p_neigh_neigh*D], &points[p*D], D);
                tmp_dist_val = euclidean_distance_sqr(p_neigh_neigh, p, points, D, N);

                if(tmp_dist_val < max_dist_val){
                    // printf("%d %d\n", knn_indices[max_id_point] ,p_neigh_neigh);
                    knn_indices[max_id_point] = p_neigh_neigh;
                    knn_sqr_dist[max_id_point] = tmp_dist_val;

                    max_id_point = knn_id;
                    max_dist_val = knn_sqr_dist[knn_id];
                    for(int j=1; j < K; ++j){
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

__global__
void nearest_neighbors_exploring_coalesced(typepoints* points,
                                 int* old_knn_indices,
                                 int* knn_indices,
                                 typepoints* knn_sqr_dist,
                                 int N, int D, int K)
{
    
    int p, tmp_p, tmp_candidate, max_id_point, p_neigh, p_neigh_neigh;
    int i,j,k;
    int knn_id;
    typepoints max_dist_val;

    __shared__ typepoints candidate_dist_val[1024];

    int tid = blockDim.x*blockIdx.x+threadIdx.x;
    int tidw = threadIdx.x % 32; // my id on warp
    int init_warp_on_block = threadIdx.x-tidw;


    for(p = tid; __any_sync(__activemask(), p < N); p+=blockDim.x*gridDim.x){
        if(p < N){
            knn_id = p*K;
            // __syncthreads();
            // __syncwarp();

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
                    candidate_dist_val[init_warp_on_block+tidw] = 0.0f;
                }

                __syncthreads();
                // tmp_dist_val = euclidean_distance_sqr(&points[p_neigh_neigh*D], &points[p*D], D);
                for(j=0; j < 32; ++j){
                    tmp_candidate = __shfl_sync(__activemask(), p_neigh_neigh, j);
                    if(tmp_candidate == -1) continue;
                    tmp_p = __shfl_sync(__activemask(), p, j);
                    euclidean_distance_sqr_coalesced(tmp_candidate, tmp_p, points, D, N,
                                                     tidw, &candidate_dist_val[init_warp_on_block+j]);
                }
                __syncwarp();
                if(p_neigh_neigh == -1) continue;

                
                if(candidate_dist_val[init_warp_on_block+tidw] < max_dist_val){
                    // printf("%d %d\n", knn_indices[max_id_point] ,p_neigh_neigh);
                    knn_indices[max_id_point] = p_neigh_neigh;
                    knn_sqr_dist[max_id_point] = candidate_dist_val[init_warp_on_block+tidw];

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