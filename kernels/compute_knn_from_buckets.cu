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
        diff = points[N*i+p1] - points[N*i+p2];
        ret += diff*diff;
    }

    return ret;
}


__global__
void compute_knn_from_buckets(int* points_parent,
                              int* points_depth,
                              int* accumulated_nodes_count,
                              typepoints* points,
                              int* node_idx_to_leaf_idx,
                              int* nodes_bucket,
                              int* count_bucket,
                              int* knn_indices,
                              typepoints* knn_sqr_dist,
                              int N, int D, int max_bucket_size, int K,
                              int MAX_TREE_CHILD, int total_buckets)
{
    int tid = blockDim.x*blockIdx.x+threadIdx.x;
    int parent_id, bucket_size, max_id_point, tmp_point;
    typepoints max_dist_val, tmp_dist_val;
    
    int knn_id;
    int wid = tid % 32; // my id in warp
    for(int bid = tid/32; bid < total_buckets; bid+=blockDim.x*gridDim.x/32){
        // __syncthreads();

        bucket_size = count_bucket[bid];
        for(int _p = wid; _p < bucket_size; _p+=32){
            // __syncthreads();
            __syncwarp();
            int p = nodes_bucket[bid*max_bucket_size + _p];
            knn_id = p*K;
            

            // TODO: Run a first scan?
            max_id_point = knn_id;
            max_dist_val = knn_sqr_dist[knn_id];

            for(int j=1; j < K; ++j){
                if(knn_sqr_dist[knn_id+j] > max_dist_val){
                    max_id_point = knn_id+j;
                    max_dist_val = knn_sqr_dist[knn_id+j];
                }
            }
            
            for(int i=0; i < bucket_size; ++i){
                __syncwarp();
                tmp_point = nodes_bucket[bid*max_bucket_size + i];
                for(int j=0; j < K; ++j){
                    if(tmp_point == knn_indices[knn_id+j]){
                        tmp_point = -1;
                        break;
                    }
                }
                if(tmp_point == -1) continue;

                tmp_dist_val = euclidean_distance_sqr(tmp_point, p, points, D, N);

                if(tmp_dist_val < max_dist_val){
                    knn_indices[max_id_point] = tmp_point;
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
void compute_knn_from_buckets_old(int* points_parent,
                              int* points_depth,
                              int* accumulated_nodes_count,
                              typepoints* points,
                              int* node_idx_to_leaf_idx,
                              int* nodes_bucket,
                              int* count_bucket,
                              int* knn_indices,
                              typepoints* knn_sqr_dist,
                              int N, int D, int max_bucket_size, int K,
                              int MAX_TREE_CHILD)
{
    int tid = blockDim.x*blockIdx.x+threadIdx.x;
    int parent_id, bucket_size, max_id_point, tmp_point;
    typepoints max_dist_val, tmp_dist_val;
    
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
        bucket_size = count_bucket[node_idx_to_leaf_idx[parent_id]];
        // for(int i=0; i < K; ++i){
        //     local_knn_indices[i] = knn_indices[p*K+i];
        //     local_knn_sqr_dist[i] = knn_sqr_dist[p*K+i];
        // }

        // TODO: Run a first scan?
        max_id_point = knn_id;
        max_dist_val = knn_sqr_dist[knn_id];

        for(int j=1; j < K; ++j){
            if(knn_sqr_dist[knn_id+j] > max_dist_val){
                // tmp_dist_val = local_knn_sqr_dist[j];
                max_id_point = knn_id+j;
                max_dist_val = knn_sqr_dist[knn_id+j];
            }
        }
        
        for(int i=0; i < bucket_size; ++i){
            __syncthreads();
            // __syncwarp();
            tmp_point = nodes_bucket[node_idx_to_leaf_idx[parent_id]*max_bucket_size + i];
            // if(p == tmp_point) continue;
            for(int j=0; j < K; ++j){
                if(tmp_point == knn_indices[knn_id+j]){
                    tmp_point = -1;
                    break;
                }
            }
            if(tmp_point == -1) continue;

            // tmp_dist_val = euclidean_distance_sqr(&points[tmp_point*D], &points[p*D], D);
            tmp_dist_val = euclidean_distance_sqr(tmp_point, p, points, D, N);

            if(tmp_dist_val < max_dist_val){
                // local_knn_indices[max_id_point] = tmp_point;
                // local_knn_sqr_dist[max_id_point] = tmp_dist_val;
                knn_indices[max_id_point] = tmp_point;
                knn_sqr_dist[max_id_point] = tmp_dist_val;

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