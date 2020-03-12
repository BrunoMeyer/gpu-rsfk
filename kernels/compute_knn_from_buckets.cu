#ifndef __COMPUTE_KNN_FROM_BUCKETS__CU
#define __COMPUTE_KNN_FROM_BUCKETS__CU


// __device__
// inline
// float euclidean_distance_sqr(typepoints* v1, typepoints* v2, int D)
// {
//     typepoints ret = 0.0f;
//     typepoints diff;

//     for(int i=0; i < D; ++i){
//         diff = v1[i] - v2[i];
//         ret += diff*diff;
//     }

//     return ret;
// }

__device__
inline
float euclidean_distance_sqr(int p1, int p2, typepoints* points, int D, int N)
{
    typepoints ret = 0.0f;
    typepoints diff;

    for(int i=0; i < D; ++i){
        diff = points[get_point_idx(p1,i,N,D)] - points[get_point_idx(p2,i,N,D)];
        ret += diff*diff;
    }

    return ret;
}


__device__
inline
float euclidean_distance_sqr_small_block(int p1, int p2, typepoints* local_points,
                                         typepoints* points, int D, int N)
{
    typepoints ret = 0.0f;
    typepoints diff;

    for(int i=0; i < D; ++i){
        diff = local_points[get_point_idx(p1,i,32,D)] - points[get_point_idx(p2,i,N,D)];
        ret += diff*diff;
    }

    return ret;
}





__device__
inline
void euclidean_distance_sqr_coalesced(int p1, int p2, typepoints* points, int D,
                                      int N, int tidw, typepoints* diff_sqd)
{
    typepoints diff;

    for(int i=tidw; i < D; i+=32){
        diff = points[get_point_idx(p1,i,N,D)] - points[get_point_idx(p2,i,N,D)];
        atomicAdd(diff_sqd,diff*diff);
    }
}


// Assign a bucket (leaf in the tree) to each warp and a point to each thread (persistent kernel)
__global__
void compute_knn_from_buckets_perwarp_coalesced(int* points_parent,
                              int* points_depth,
                              int* accumulated_nodes_count,
                              typepoints* points,
                              int* node_idx_to_leaf_idx,
                              int* nodes_bucket,
                              int* bucket_size,
                              int* knn_indices,
                              typepoints* knn_sqr_dist,
                              int N, int D, int max_bucket_size, int K,
                              int MAX_TREE_CHILD, int total_buckets)
{
    int tid = blockDim.x*blockIdx.x+threadIdx.x;
    int parent_id, current_bucket_size, max_id_point, candidate_point;
    typepoints max_dist_val, candidate_dist_val;
    
    int knn_id;
    int tidw = threadIdx.x % 32; // my id on warp
    int init_warp_on_block = threadIdx.x-tidw;
    extern __shared__ typepoints local_candidate_dist_val[];
    
    for(int bid = tid/32; bid < total_buckets; bid+=blockDim.x*gridDim.x/32){
        __syncthreads();
        // __syncwarp();
        current_bucket_size = bucket_size[bid];
        for(int _p = tidw; __any_sync(__activemask(),_p < current_bucket_size); _p+=32){
            // __syncthreads();
            int p;
            if(_p < current_bucket_size){
                p = nodes_bucket[bid*max_bucket_size + _p];
                knn_id = p*K;
                parent_id = accumulated_nodes_count[points_depth[p]] + points_parent[p];
                current_bucket_size = bucket_size[node_idx_to_leaf_idx[parent_id]];

                max_id_point = knn_id;
                max_dist_val = knn_sqr_dist[knn_id];
                // Finds the index of the furthest point from the current result of knn_indices
                // and the distance between them
                for(int j=1; j < K; ++j){
                    if(knn_sqr_dist[knn_id+j] > max_dist_val){
                        max_id_point = knn_id+j;
                        max_dist_val = knn_sqr_dist[knn_id+j];
                    }
                }
            }

            // __syncwarp();
            for(int i=0; i < current_bucket_size; ++i){
                __syncthreads();
                candidate_point = -1;
                if(_p < current_bucket_size){
                    candidate_point = nodes_bucket[node_idx_to_leaf_idx[parent_id]*max_bucket_size + i];
                    
                    // Verify if the candidate point (inside the bucket of current point)
                    // already is in the knn_indices result
                    for(int j=0; j < K; ++j){
                        if(candidate_point == knn_indices[knn_id+j]){
                            candidate_point = -1;
                            break;
                        }
                    }

                    // If it is, then it doesnt need to be treated, then go to
                    // the next iteration and wait the threads from same warp to goes on

                    local_candidate_dist_val[threadIdx.x] = 0.0;
                }
                int tmp_candidate;
                int tmp_p;
                // local_candidate_dist_val[threadIdx.x] = euclidean_distance_sqr(candidate_point, p, points, D, N);
                __syncwarp();
                for(int j=0; j < 32; ++j){
                    tmp_candidate = __shfl_sync(__activemask(), candidate_point, j);
                    if(tmp_candidate == -1) continue;
                    tmp_p = __shfl_sync(__activemask(), p, j);
                    euclidean_distance_sqr_coalesced(tmp_candidate, tmp_p, points, D, N,
                                                    tidw,
                                                    &local_candidate_dist_val[init_warp_on_block+j]);
                }

                if(candidate_point == -1) continue;

                // If the candidate is closer than the pre-computed furthest point,
                // switch them
                if(local_candidate_dist_val[init_warp_on_block+tidw] < max_dist_val){
                    // local_knn_indices[max_id_point] = candidate_point;
                    // local_knn_sqr_dist[max_id_point] = candidate_dist_val;
                    knn_indices[max_id_point] = candidate_point;
                    knn_sqr_dist[max_id_point] = local_candidate_dist_val[init_warp_on_block+tidw];

                    // Also update the furthest point that will be used in the next
                    // comparison
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
        __syncwarp();
        }
    }
}

// Assign a bucket (leaf in the tree) to each warp and a point to each thread (persistent kernel)
__global__
void compute_knn_from_buckets_coalesced(int* points_parent,
                              int* points_depth,
                              int* accumulated_nodes_count,
                              typepoints* points,
                              int* node_idx_to_leaf_idx,
                              int* nodes_bucket,
                              int* bucket_size,
                              int* knn_indices,
                              typepoints* knn_sqr_dist,
                              int N, int D, int max_bucket_size, int K,
                              int MAX_TREE_CHILD)
{
    int tid = blockDim.x*blockIdx.x+threadIdx.x;
    int parent_id, current_bucket_size, max_id_point, candidate_point;
    
    typepoints max_dist_val;
    // TODO: Accept others block size rather than 1024

    __shared__ typepoints candidate_dist_val[1024/32];

    for(int i=threadIdx.x; i < 1024/32; ++i){
        candidate_dist_val[i] = 0.0;
    }
    __syncthreads();

    int knn_id;

    int tidw = threadIdx.x % 32; // my id on warp
    int wid =  threadIdx.x/32; // warp id


    
    for(int p = tid/32; p < N; p+=blockDim.x*gridDim.x/32){
        knn_id = p*K;

        parent_id = accumulated_nodes_count[points_depth[p]] + points_parent[p];
        current_bucket_size = bucket_size[node_idx_to_leaf_idx[parent_id]];

        // TODO: Run a first scan?
        max_id_point = knn_id;
        max_dist_val = knn_sqr_dist[knn_id];
        
        // Finds the index of the furthest point from the current result of knn_indices
        // and the distance between them
        for(int j=1; j < K; ++j){
            if(knn_sqr_dist[knn_id+j] > max_dist_val){
                max_id_point = knn_id+j;
                max_dist_val = knn_sqr_dist[knn_id+j];
            }
        }

        
        for(int i=0; i < current_bucket_size; ++i){
            __syncthreads();
            __syncwarp();
            candidate_point = nodes_bucket[node_idx_to_leaf_idx[parent_id]*max_bucket_size + i];
            
            // Verify if the candidate point (inside the bucket of current point)
            // already is in the knn_indices result
            for(int j=0; j < K; ++j){
                if(candidate_point == knn_indices[knn_id+j]){
                    candidate_point = -1;
                    break;
                }
            }

            // If it is, then it doesnt need to be treated, then go to
            // the next iteration and wait the threads from same warp to goes on
            if(candidate_point == -1) continue;

            euclidean_distance_sqr_coalesced(candidate_point, p, points, D, N,
                                             tidw, &candidate_dist_val[wid]);

            __syncwarp(); // This is fundamental once that all threads
                          // in the warp are calculating the distance

            // If the candidate is closer than the pre-computed furthest point,
            // switch them
            if(tidw == 0){
                if(candidate_dist_val[wid] < max_dist_val){
                    // printf("%f %f %f\n", candidate_dist_val[wid], knn_sqr_dist[max_id_point], max_dist_val);

                    knn_indices[max_id_point] = candidate_point;
                    knn_sqr_dist[max_id_point] = candidate_dist_val[wid];

                    // Also update the furthest point that will be used in the next
                    // comparison
                    max_id_point = knn_id;
                    max_dist_val = knn_sqr_dist[knn_id];
                    for(int j=1; j < K; ++j){
                        if(knn_sqr_dist[knn_id+j] > max_dist_val){
                            max_id_point = knn_id+j;
                            max_dist_val = knn_sqr_dist[knn_id+j];
                        }
                    }
                }
                candidate_dist_val[wid] = 0.0;
            }
            __syncwarp();
        }
    }
}





// Assumes that each block contains only 32 (warp size) threads (persistent kernel)
__global__
void compute_knn_from_buckets_small_block(int* points_parent,
                              int* points_depth,
                              int* accumulated_nodes_count,
                              typepoints* points,
                              int* node_idx_to_leaf_idx,
                              int* nodes_bucket,
                              int* bucket_size,
                              int* knn_indices,
                              typepoints* knn_sqr_dist,
                              int N, int D, int max_bucket_size, int K,
                              int MAX_TREE_CHILD, int total_buckets)
{
    int tid = blockDim.x*blockIdx.x+threadIdx.x;
    int parent_id, current_bucket_size, max_id_point, candidate_point;
    typepoints max_dist_val, candidate_dist_val;
    
    int knn_id;
    int wid = tid;

    // extern __shared__ int local_knn_indices[];
    // extern __shared__ typepoints local_knn_sqr_dist[];
    // extern __shared__ typepoints local_points[];
    

    for(int bid = blockIdx.x; bid < total_buckets; bid+=gridDim.x){
        __syncthreads();
        // __syncwarp();
        current_bucket_size = bucket_size[bid];
        for(int _p = threadIdx.x; _p < current_bucket_size; _p+=32){
            // __syncthreads();
            __syncwarp();
            int p = nodes_bucket[bid*max_bucket_size + _p];
            knn_id = p*K;
            
            max_id_point = knn_id;
            max_dist_val = knn_sqr_dist[knn_id];

            // Finds the index of the furthest point from the current result of knn_indices
            // and the distance between them
            for(int j=1; j < K; ++j){
                // __syncwarp();
                if(knn_sqr_dist[knn_id+j] > max_dist_val){
                    max_id_point = knn_id+j;
                    max_dist_val = knn_sqr_dist[knn_id+j];
                }
            }

            // for(int i=0; i < D; ++i){
            //     local_points[32*i+threadIdx.x] = points[N*i+p];
            // }
            // __syncwarp();
            __syncthreads();

            for(int i=0; i < current_bucket_size; ++i){
                // __syncwarp();
                candidate_point = nodes_bucket[bid*max_bucket_size + i];
                // Verify if the candidate point (inside the bucket of current point)
                // already is in the knn_indices result
                for(int j=0; j < K; ++j){
                    // __syncwarp();
                    if(candidate_point == knn_indices[knn_id+j]){
                        candidate_point = -1;
                        break;
                    }
                }

                // If it is, then it doesnt need to be treated, then go to
                // the next iteration and wait the threads from same warp to goes on
                if(candidate_point == -1) continue;

                // __syncwarp();
                // printf("%d %d\n",p, candidate_point);

                // candidate_dist_val = euclidean_distance_sqr_small_block(threadIdx.x, candidate_point, local_points, points, D, N);
                candidate_dist_val = euclidean_distance_sqr(candidate_point, p, points, D, N);

                // If the candidate is closer than the pre-computed furthest point,
                // switch them
                if(candidate_dist_val < max_dist_val){
                    knn_indices[max_id_point] = candidate_point;
                    knn_sqr_dist[max_id_point] = candidate_dist_val;

                    // Also update the furthest point that will be used in the next
                    // comparison
                    max_id_point = knn_id;
                    max_dist_val = knn_sqr_dist[knn_id];
                    for(int j=1; j < K; ++j){
                        // __syncwarp();
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


// Assign a bucket (leaf in the tree) to each warp and a point to each thread (persistent kernel)
__global__
void compute_knn_from_buckets(int* points_parent,
                              int* points_depth,
                              int* accumulated_nodes_count,
                              typepoints* points,
                              int* node_idx_to_leaf_idx,
                              int* nodes_bucket,
                              int* bucket_size,
                              int* knn_indices,
                              typepoints* knn_sqr_dist,
                              int N, int D, int max_bucket_size, int K,
                              int MAX_TREE_CHILD, int total_buckets)
{
    int tid = blockDim.x*blockIdx.x+threadIdx.x;
    int parent_id, current_bucket_size, max_id_point, candidate_point;
    typepoints max_dist_val, candidate_dist_val;
    
    int knn_id;
    int wid = tid % 32; // my id on warp
    for(int bid = tid/32; bid < total_buckets; bid+=blockDim.x*gridDim.x/32){
        __syncthreads();
        // __syncwarp();
        current_bucket_size = bucket_size[bid];
        for(int _p = wid; _p < current_bucket_size; _p+=32){
            // __syncthreads();
            __syncwarp();
            int p = nodes_bucket[bid*max_bucket_size + _p];
            knn_id = p*K;
            
            max_id_point = knn_id;
            max_dist_val = knn_sqr_dist[knn_id];

            // Finds the index of the furthest point from the current result of knn_indices
            // and the distance between them
            for(int j=1; j < K; ++j){
                // __syncwarp();
                if(knn_sqr_dist[knn_id+j] > max_dist_val){
                    max_id_point = knn_id+j;
                    max_dist_val = knn_sqr_dist[knn_id+j];
                }
            }
            
            for(int i=0; i < current_bucket_size; ++i){
                // __syncwarp();

                // Verify if the candidate point (inside the bucket of current point)
                // already is in the knn_indices result
                candidate_point = nodes_bucket[bid*max_bucket_size + i];
                for(int j=0; j < K; ++j){
                    // __syncwarp();
                    if(candidate_point == knn_indices[knn_id+j]){
                        candidate_point = -1;
                        break;
                    }
                }

                // If it is, then it doesnt need to be treated, then go to
                // the next iteration and wait the threads from same warp to goes on
                if(candidate_point == -1) continue;

                // __syncwarp();
                candidate_dist_val = euclidean_distance_sqr(candidate_point, p, points, D, N);


                // If the candidate is closer than the pre-computed furthest point,
                // switch them
                if(candidate_dist_val < max_dist_val){
                    knn_indices[max_id_point] = candidate_point;
                    knn_sqr_dist[max_id_point] = candidate_dist_val;

                    // Also update the furthest point that will be used in the next
                    // comparison
                    max_id_point = knn_id;
                    max_dist_val = knn_sqr_dist[knn_id];
                    for(int j=1; j < K; ++j){
                        // __syncwarp();
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


// Assign each point to a thread (persistent kernel)
__global__
void compute_knn_from_buckets_old(int* points_parent,
                              int* points_depth,
                              int* accumulated_nodes_count,
                              typepoints* points,
                              int* node_idx_to_leaf_idx,
                              int* nodes_bucket,
                              int* bucket_size,
                              int* knn_indices,
                              typepoints* knn_sqr_dist,
                              int N, int D, int max_bucket_size, int K,
                              int MAX_TREE_CHILD)
{
    int tid = blockDim.x*blockIdx.x+threadIdx.x;
    int parent_id, current_bucket_size, max_id_point, candidate_point;
    typepoints max_dist_val, candidate_dist_val;
    
    int knn_id;
    // extern __shared__ int local_knn_indices[];
    // extern __shared__ typepoints local_knn_sqr_dist[];
    
    // local_knn_indices = &local_knn_indices[MAX_TREE_CHILD*threadIdx.x];
    // local_knn_sqr_dist = &local_knn_sqr_dist[MAX_TREE_CHILD*threadIdx.x];

    for(int p = tid; p < N; p+=blockDim.x*gridDim.x){
        knn_id = p*K;
        // __syncthreads();
        // __syncwarp();
        parent_id = accumulated_nodes_count[points_depth[p]] + points_parent[p];
        current_bucket_size = bucket_size[node_idx_to_leaf_idx[parent_id]];
        // for(int i=0; i < K; ++i){
        //     local_knn_indices[i] = knn_indices[p*K+i];
        //     local_knn_sqr_dist[i] = knn_sqr_dist[p*K+i];
        // }

        // TODO: Run a first scan?
        max_id_point = knn_id;
        max_dist_val = knn_sqr_dist[knn_id];
        
        // Finds the index of the furthest point from the current result of knn_indices
        // and the distance between them
        for(int j=1; j < K; ++j){
            if(knn_sqr_dist[knn_id+j] > max_dist_val){
                // candidate_dist_val = local_knn_sqr_dist[j];
                max_id_point = knn_id+j;
                max_dist_val = knn_sqr_dist[knn_id+j];
            }
        }
        
        for(int i=0; i < current_bucket_size; ++i){
            __syncthreads();
            // __syncwarp();
            candidate_point = nodes_bucket[node_idx_to_leaf_idx[parent_id]*max_bucket_size + i];
            // if(p == candidate_point) continue;
            
            // Verify if the candidate point (inside the bucket of current point)
            // already is in the knn_indices result
            for(int j=0; j < K; ++j){
                if(candidate_point == knn_indices[knn_id+j]){
                    candidate_point = -1;
                    break;
                }
            }

            // If it is, then it doesnt need to be treated, then go to
            // the next iteration and wait the threads from same warp to goes on
            if(candidate_point == -1) continue;

            // candidate_dist_val = euclidean_distance_sqr(&points[candidate_point*D], &points[p*D], D);
            candidate_dist_val = euclidean_distance_sqr(candidate_point, p, points, D, N);

            // If the candidate is closer than the pre-computed furthest point,
            // switch them
            if(candidate_dist_val < max_dist_val){
                // local_knn_indices[max_id_point] = candidate_point;
                // local_knn_sqr_dist[max_id_point] = candidate_dist_val;
                knn_indices[max_id_point] = candidate_point;
                knn_sqr_dist[max_id_point] = candidate_dist_val;

                // Also update the furthest point that will be used in the next
                // comparison
                max_id_point = knn_id;
                max_dist_val = knn_sqr_dist[knn_id];
                for(int j=1; j < K; ++j){
                    // if(local_knn_sqr_dist[j] > max_dist_val){
                    if(knn_sqr_dist[knn_id+j] > max_dist_val){
                        max_id_point = knn_id+j;
                        // max_dist_val = local_knn_sqr_dist[j];
                        max_dist_val = knn_sqr_dist[knn_id+j];
                    }
                }
            }
        }
        // __syncthreads();
        // for(int i=0; i < K; ++i){
        //     knn_indices[p*K+i]  = local_knn_indices[i];
        //     knn_sqr_dist[p*K+i] = local_knn_sqr_dist[i];
        // }
    }
}

#endif