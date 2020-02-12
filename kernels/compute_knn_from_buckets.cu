#ifndef __COMPUTE_KNN_FROM_BUCKETS__CU
#define __COMPUTE_KNN_FROM_BUCKETS__CU


__device__
inline
float euclidean_distance_sqr(typepoints* v1, typepoints* v2, int D)
{
    typepoints ret = 0.0f;
    typepoints diff;

    for(int i=0; i < D; ++i){
        diff = v1[i] - v2[i];
        ret += diff*diff;
    }

    return ret;
}

__global__
void compute_knn_from_buckets(int* points_parent,
                              int* points_depth,
                              int* accumulated_nodes_count,
                              int* child_count,
                              typepoints* points,
                              int* bucket_nodes,
                              int* knn_indices,
                              typepoints* knn_sqr_dist,
                              int N, int D, int max_bucket_size, int K)
{
    int tid = blockDim.x*blockIdx.x+threadIdx.x;
    int parent_id, bucket_size, max_id_point, tmp_point;
    typepoints max_dist_val, tmp_dist_val;
    
    int local_knn_indices[MAX_TREE_CHILD];
    typepoints local_knn_sqr_dist[MAX_TREE_CHILD];

    for(int p = tid; p < N; p+=blockDim.x*gridDim.x){
        parent_id = accumulated_nodes_count[points_depth[p]] + points_parent[p];
        bucket_size = child_count[parent_id];
        for(int i=0; i < K; ++i){
            local_knn_indices[i] = knn_indices[p*K+i];
            local_knn_sqr_dist[i] = knn_sqr_dist[p*K+i];
        }
        // __syncthreads();

        // TODO: Run a first scan?
        max_id_point = 0;
        max_dist_val = local_knn_sqr_dist[0];

        // for(int j=1; j < K; ++j){
        //     if(local_knn_sqr_dist[j] > max_dist_val){
        //         // tmp_dist_val = local_knn_sqr_dist[j];
        //         max_id_point = j;
        //         max_dist_val = local_knn_sqr_dist[j];
        //     }
        // }
        
        for(int i=0; i < bucket_size; ++i){
            tmp_point = bucket_nodes[max_bucket_size*parent_id + i];
            if(p == tmp_point) continue;

            tmp_dist_val = euclidean_distance_sqr(&points[tmp_point], &points[p], D);
            if(tmp_dist_val < max_dist_val){
                local_knn_indices[max_id_point] = tmp_point;
                local_knn_sqr_dist[max_id_point] = tmp_dist_val;

                max_dist_val = tmp_dist_val;
                for(int j=0; j < K; ++j){
                    if(local_knn_sqr_dist[j] > max_dist_val){
                        max_id_point = j;
                        max_dist_val = local_knn_sqr_dist[j];
                    }
                }
            }
        }
        for(int i=0; i < K; ++i){
            knn_indices[p*K+i]  = local_knn_indices[i];
            knn_sqr_dist[p*K+i] = local_knn_sqr_dist[i];
        }
    }
}

#endif