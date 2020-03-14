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

#endif