#include "include/rsfk.h"
#include "knng-yykmeans.h"
#include "kmeans/kernel-functions/gpu-utils.cu"
#include "kmeans/cpu-utils.c"

#define SQRT_FUNC(x) __fsqrt_rn(x)
// #define SQRT_FUNC(x) sqrt(x)

__global__
void compute_bucket_radii_persistent(
    float* sqr_dist_to_cents,
    uint* labels_buckets,
    float* bucket_radii,
    int N, int total_buckets
){
    extern __shared__ float shared_mem[];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    //filling shared memory with zeros
    for(int i = threadIdx.x; i < total_buckets; i += blockDim.x){
        shared_mem[i] = 0.0f;
    }
    // Each thread processes multiple points in the bucket
    for(int i = tid; i < N; i += blockDim.x*gridDim.x){
        float dist = sqr_dist_to_cents[i];
        int bucket_id = labels_buckets[i];
        if(dist > shared_mem[bucket_id]){
            atomicMaxFloat(&shared_mem[bucket_id], dist);  
        }
    }
    __syncthreads();
    for(int i = threadIdx.x; i < total_buckets; i += blockDim.x){
        atomicMaxFloat(&bucket_radii[i], SQRT_FUNC(shared_mem[i]));
    }
}

__global__
void compute_bucket_radii_not_persistent(
    float* sqr_dist_to_cents,
    uint* labels_buckets,
    float* bucket_radii,
    int N, int total_buckets
){
    extern __shared__ float shared_mem[];
    int bucket_id = blockIdx.x;
    int tid = threadIdx.x;
    float max_sqrd_radius = 0.0f;

    // Each thread processes multiple points in the bucket
    for(int i = tid; i < N; i += blockDim.x){
        if(labels_buckets[i] == bucket_id){
            float dist = sqr_dist_to_cents[i];
            if(dist > max_sqrd_radius){
                max_sqrd_radius = dist;
            }
        }
    }

    // Store the maximum squared radius found by this thread in shared memory
    shared_mem[tid] = max_sqrd_radius;
    __syncthreads();

    // Reduce within the block to find the maximum squared radius
    for(int s = blockDim.x / 2; s > 0; s >>= 1){
        if(tid < s){
            if(shared_mem[tid + s] > shared_mem[tid]){
                shared_mem[tid] = shared_mem[tid + s];
            }
        }
        __syncthreads();
    }

    // The first thread writes the result to global memory
    if(tid == 0){
        bucket_radii[bucket_id] = SQRT_FUNC(shared_mem[0]);
    }
}

__global__
void bucket_exploring_kernel(
    RSFK_typepoints* points,
    int* knn_indices,
    RSFK_typepoints* knn_sqr_dist,
    uint *kmeans_labels,
    float *bucket_centroids,
    float *sqr_dist_to_cents,
    float *bucket_radii,
    int* nodes_buckets,
    int* bucket_sizes,
    int total_buckets,
    int max_bucket_size,
    int K, int N, int D
    #if COUNT_FILTER_EFFECTIVENESS
    , int global_n_filters_skips[],
    int global_n_filters_calc_avoided[],
    int global_total_test[]
    #endif
){
    // #if DEBUG_BUCKET_EXPLORING2
    //     //print sqr distances to centroids
    //     if(threadIdx.x == 0 && blockIdx.x == 3){
    //         printf("DEBUG_BUCKET_EXPLORING in %s %d: sqr_dist_to_cents: [", __FILE__,__LINE__);
    //         for(int i=0; i<N; i++){
    //             printf("%f, ", sqr_dist_to_cents[i]);
    //         }
    //         printf("]\n");
    //     }
    //     // return;
    // #endif


    // TODO: CHECK SM MEMORY USAGE
    __shared__ RSFK_typepoints candidate_dist_sm[1024];
    __shared__ RSFK_typepoints candidate_idx_sm[1024];
    __shared__ int n_candidates_sm;

    __shared__ int sm_point_id[1024];
    __shared__ RSFK_typepoints max_neighbor_sqr_dist_sm[1024];
    __shared__ int max_neighbor_pos[1024];
    __shared__ RSFK_typepoints max_neigh_dist_bucket_sm; // to hold point coordinates
    __shared__ RSFK_typepoints bcenters_dist_sm[32];

    __shared__ int bucket_to_explore_sm; 
    __shared__ int point_to_explore_sm; 
   
    int wid = threadIdx.x / 32; // my warp id
    int lane = threadIdx.x % 32; // my id on warp
    int nwarps = blockDim.x / 32; // number of warps per block

    int bsize = bucket_sizes[blockIdx.x]; // current bucket size
    // reference bucket radius
    RSFK_typepoints rb_radius = bucket_radii[blockIdx.x];

    #if COUNT_FILTER_EFFECTIVENESS
        // int n_filter1_skips=0;
        // int n_filter1_calc_avoided=0;
        // int n_filter2_skips=0;
        // int n_filter2_calc_avoided=0;
        // int n_filter3_skips=0;
        // int n_filter3_calc_avoided=0;
        // int n_filter4_skips=0;
        // int n_filter4_calc_avoided=0;

        int total_filter_tests[4] = {0,0,0,0};
    #endif

    if(lane == 0){
        max_neigh_dist_bucket_sm = -1.0f;
        bucket_to_explore_sm = -1;
        point_to_explore_sm = -1;
        n_candidates_sm = 0;
    }
    __syncthreads();
    // -----------------------------------------
    // step 1: find farthest candidate to knn graph for each point in the bucket
    for(int i=wid; i < bsize; i+=nwarps){
        int p1 = nodes_buckets[blockIdx.x*max_bucket_size + i];
        sm_point_id[i] = p1;
        int knn_id = p1*K; 

        RSFK_typepoints local_max_position = -1;
        RSFK_typepoints local_max_dist = -1.0f;
        for(int j=lane; j < K; j+=32){
            if(knn_sqr_dist[knn_id+j] > local_max_dist){
                local_max_position = knn_id+j;
                local_max_dist = knn_sqr_dist[knn_id+j];
            }
        }
        
        for(int offset = 16; offset > 0; offset /= 2){
            RSFK_typepoints tmp_max_position = __shfl_down_sync( 0xffffffff, local_max_position, offset); // assuming warpSize=32
            RSFK_typepoints tmp_max_dist = __shfl_down_sync( 0xffffffff, local_max_dist, offset); // assuming warpSize=32
            if(tmp_max_dist > local_max_dist){
                local_max_dist = tmp_max_dist;
                local_max_position = tmp_max_position;
            }
        }

        if(lane == 0){
            max_neighbor_sqr_dist_sm[i] = local_max_dist;
            max_neighbor_pos[i] = local_max_position;
            atomicMaxFloat(&max_neigh_dist_bucket_sm, SQRT_FUNC(local_max_dist));
        }

    }
    __syncthreads();

    // -----------------------------------------
    // step 2: explore bucket to find new candidates
    for(int ii = 0; ii < total_buckets; ii+=nwarps){
        int i = ii + wid;
        int bte = -1;
        if(i < total_buckets && i != blockIdx.x){
                RSFK_typepoints bcenter_dist = euclidean_distance_sqr(
                    &bucket_centroids[i*D],
                    &bucket_centroids[blockIdx.x*D],
                    D,lane);
            
            bcenter_dist = SQRT_FUNC(bcenter_dist);

            // FILTER 1: check if the bucket is close enough to be explored
            RSFK_typepoints pb_radius = bucket_radii[i];

            #if COUNT_FILTER_EFFECTIVENESS
                total_filter_tests[0]++;
            #endif

            #if DEBUG_BUCKET_EXPLORING
                #define CHECK_BID 0
                if(lane == 0 && blockIdx.x == CHECK_BID){
                    printf("DEBUG_BUCKET_EXPLORING in %s %d: Bucket %d center: [%f, %f, %f, %f,...], Bucket %d center: [%f, %f, %f, %f,...], Bucket %d center distance to bucket %d: %f, radius sum: %f(PB) + %f(RB) + %f(MN) = %f\n", 
                        __FILE__,__LINE__, 
                        i, bucket_centroids[i*D], bucket_centroids[i*D+1], bucket_centroids[i*D+2], bucket_centroids[i*D+3],
                        blockIdx.x, bucket_centroids[blockIdx.x*D], bucket_centroids[blockIdx.x*D+1], bucket_centroids[blockIdx.x*D+2], bucket_centroids[blockIdx.x*D+3],
                        i, blockIdx.x, bcenter_dist,
                        pb_radius, rb_radius, max_neigh_dist_bucket_sm, pb_radius + rb_radius + max_neigh_dist_bucket_sm);
                }
            #endif
            if(bcenter_dist <= pb_radius + rb_radius + max_neigh_dist_bucket_sm){
                // #if DEBUG_BUCKET_EXPLORING
                //     if(lane == 0)
                //         printf("DEBUG_BUCKET_EXPLORING in %s %d: Not skipping bucket %d for bucket %d\n", __FILE__,__LINE__, i, blockIdx.x);
                // #endif
                //don't skip this bucket
                bte = i; // bte = bucket to explore
                if(lane == 0)
                    bcenters_dist_sm[wid] = bcenter_dist;
            } 
            #if COUNT_FILTER_EFFECTIVENESS
                else {
                    if(lane == 0){
                        // n_filter1_skips++;
                        // n_filter1_calc_avoided += bucket_sizes[i]*bsize;
                        atomicAdd(&global_n_filters_skips[0], 1);
                        atomicAdd(&global_n_filters_calc_avoided[0], bucket_sizes[i]*bsize);
                    }
                }
            #endif
        }
        if(lane == 0)
            atomicMax(&bucket_to_explore_sm, bte);
        __syncthreads();
        while(bucket_to_explore_sm != -1){
            int pb = bucket_to_explore_sm; //pb = probe bucket
            if(bte == pb)
                bte = -1;

            #if DEBUG_BUCKET_EXPLORING
                if(threadIdx.x == 0)
                    printf("DEBUG_BUCKET_EXPLORING in %s %d: Exploring bucket %d for bucket %d\n", __FILE__,__LINE__, pb, blockIdx.x );
            #endif

            RSFK_typepoints pb_radius = bucket_radii[pb];
            RSFK_typepoints bcenter_dist = bcenters_dist_sm[pb % nwarps];
            RSFK_typepoints intersec = rb_radius + pb_radius - bcenter_dist;

            for(int jj= 0; jj < bsize; jj+=nwarps){
                int pte = jj + wid; // pte = point to explore
                float max_dist_pte = SQRT_FUNC(max_neighbor_sqr_dist_sm[pte]);
                if(pte < bsize){
                    int real_p1 = nodes_buckets[blockIdx.x*max_bucket_size + pte];
                    // FILTER 2: miolo of the reference bucket
                    #if COUNT_FILTER_EFFECTIVENESS
                        total_filter_tests[1]++;
                    #endif
                    #if DEBUG_BUCKET_EXPLORING2
                        if(real_p1 < 0 || real_p1 >= N){
                            if(threadIdx.x == 0)
                                printf("ERROR in %s %d: real_p1 index %d out of bounds for points array of size %d\n", __FILE__,__LINE__, real_p1, N);
                        } else {
                            if(threadIdx.x == 0 && blockIdx.x == 3)
                                printf("DEBUG_BUCKET_EXPLORING in %s %d: Evaluating point %d in bucket %d\n", __FILE__,__LINE__, real_p1, blockIdx.x );
                        }
                    #endif
                    float dist_to_cent_p1 = SQRT_FUNC(sqr_dist_to_cents[real_p1]);
                    #if DEBUG_BUCKET_EXPLORING
                        if(lane == 0 && blockIdx.x == 3){
                            printf("DEBUG_BUCKET_EXPLORING in %s %d: Point %d distance to its bucket centroid: %f, max neighbor dist: %f, rb_radius: %f, intersec: %f\n rb_radius - intersec=%f dist_to_cent_p1 + max_dist_pte=%f\n", 
                                __FILE__,__LINE__, 
                                real_p1, dist_to_cent_p1, max_dist_pte, rb_radius, intersec, rb_radius - intersec, dist_to_cent_p1 + max_dist_pte);
                        }
                    #endif
                    if(rb_radius - intersec > dist_to_cent_p1 + max_dist_pte  ){
                        //skip this point
                        pte = -1;
                        #if COUNT_FILTER_EFFECTIVENESS
                            // n_filter2_skips++;
                            // n_filter2_calc_avoided += bucket_sizes[pb];
                            atomicAdd(&global_n_filters_skips[1], 1);
                            atomicAdd(&global_n_filters_calc_avoided[1], bucket_sizes[pb]);
                        #endif
                        #if DEBUG_BUCKET_EXPLORING
                            // if(lane == 0)
                            //     printf("DEBUG_BUCKET_EXPLORING in %s %d: Skipping point %d in bucket %d due to filter 2\n", __FILE__,__LINE__, real_p1, blockIdx.x );
                        #endif

                    }
                    else {
                        // FILTER 3: check distance of the point to the probe bucket centroid
                        #if COUNT_FILTER_EFFECTIVENESS
                            total_filter_tests[2]++;
                        #endif
                        RSFK_typepoints p1_pb_centroid_dist = euclidean_distance_sqr(
                            &points[real_p1*D],
                            &bucket_centroids[pb*D],
                            D,lane);
                        p1_pb_centroid_dist = SQRT_FUNC(p1_pb_centroid_dist);
                        #if DEBUG_BUCKET_EXPLORING
                            if(lane == 0 && blockIdx.x == CHECK_BID){
                                printf("DEBUG_BUCKET_EXPLORING in %s %d: Point %d distance to probe bucket %d centroid: %f, pb_radius: %f, max_dist_pte: %f, filter: (p1_pb_centroid_dist > pb_radius + max_dist_pte)\n", 
                                    __FILE__,__LINE__, 
                                    real_p1, pb, p1_pb_centroid_dist, pb_radius, max_dist_pte);
                            }
                        #endif
                        if( p1_pb_centroid_dist > pb_radius + max_dist_pte ){
                            //skip this point
                            pte = -1;
                            #if COUNT_FILTER_EFFECTIVENESS
                                // n_filter3_skips++;
                                // n_filter3_calc_avoided += bucket_sizes[pb];
                                atomicAdd(&global_n_filters_skips[2], 1);
                                atomicAdd(&global_n_filters_calc_avoided[2], bucket_sizes[pb]);
                            #endif
                            #if DEBUG_BUCKET_EXPLORING
                                // if(lane == 0)
                                //     printf("DEBUG_BUCKET_EXPLORING in %s %d: Skipping point %d in bucket %d due to filter 3\n", __FILE__,__LINE__, real_p1, blockIdx.x );
                            #endif
                        }
                    }
                } else{
                    pte = -1;
                }
                if(lane == 0)
                    atomicMax(&point_to_explore_sm, pte);
                __syncthreads();
                // #if DEBUG_BUCKET_EXPLORING
                //     if(threadIdx.x == 0)
                //         printf("DEBUG_BUCKET_EXPLORING in %s %d: First point to explore in bucket %d is %d\n", __FILE__,__LINE__, pb, point_to_explore_sm );
                // #endif

                
                while(point_to_explore_sm != -1){
                    int p1 = point_to_explore_sm; // pte = point to explore
                    if( p1 == pte)
                        pte = -1;
                    #if DEBUG_BUCKET_EXPLORING2
                        if(p1 < 0 || p1 >= max_bucket_size){
                            if(threadIdx.x == 0)
                                printf("ERROR in %s %d: p1 index %d out of bounds for bucket size %d\n", __FILE__,__LINE__, p1, bsize);
                            break;
                        } else {
                            if(threadIdx.x == 0 && blockIdx.x == 3)
                                printf("DEBUG_BUCKET_EXPLORING in %s %d: Evaluating point %d in bucket %d\n", __FILE__,__LINE__, 
                                    nodes_buckets[blockIdx.x*max_bucket_size + p1], blockIdx.x);
                        }
                    #endif
                    int real_p1 = nodes_buckets[blockIdx.x*max_bucket_size + p1];
                    float max_dist_p1 = SQRT_FUNC(max_neighbor_sqr_dist_sm[p1]);

                    // #if DEBUG_BUCKET_EXPLORING
                    //     if(lane == 0)
                    //         printf("DEBUG_BUCKET_EXPLORING in %s %d: Not skipping point %d in bucket %d\n", __FILE__,__LINE__, real_p1, blockIdx.x );
                    // #endif

                    // after all filters passed, explore the probe bucket
                    for(int p2=wid; p2 < bucket_sizes[pb]; p2+=nwarps){
                        // FILTER 4: miolo of the probe bucket
                        #if COUNT_FILTER_EFFECTIVENESS
                            total_filter_tests[3]++;
                        #endif
                        #if DEBUG_BUCKET_EXPLORING2
                            if(p2 < 0 || p2 >= max_bucket_size){
                                if(threadIdx.x == 0)
                                    printf("ERROR in %s %d: p2 index %d out of bounds for bucket size %d\n", __FILE__,__LINE__, p2, bucket_sizes[pb]);
                                break;
                            } else {
                                if(threadIdx.x == 0 && blockIdx.x == 3)
                                    printf("DEBUG_BUCKET_EXPLORING in %s %d: Evaluating point %d in bucket %d for point %d in bucket %d\n", __FILE__,__LINE__, 
                                        nodes_buckets[pb*max_bucket_size + p2], pb,
                                        real_p1, blockIdx.x);
                            }
                        #endif
                        int real_p2 = nodes_buckets[pb*max_bucket_size + p2];
                        float dist_to_cent_realp2 = SQRT_FUNC(sqr_dist_to_cents[real_p2]);
                        if(dist_to_cent_realp2 + max_dist_p1 < pb_radius - intersec){
                            //skip this point
                            #if COUNT_FILTER_EFFECTIVENESS
                                // n_filter4_skips++;
                                // n_filter4_calc_avoided++;
                                atomicAdd(&global_n_filters_skips[3], 1);
                                atomicAdd(&global_n_filters_calc_avoided[3], 1);
                            #endif
                            continue;
                        }

                        // compute distance
                        // #if DEBUG_BUCKET_EXPLORING
                        //     if(lane == 0)
                        //         printf("DEBUG_BUCKET_EXPLORING in %s %d: Not skipping distance calculation for points %d and %d\n", __FILE__,__LINE__, real_p1, real_p2 );
                        // #endif
                        RSFK_typepoints p1_p2_dist = euclidean_distance_sqr(
                            &points[real_p1*D],
                            &points[real_p2*D],
                            D,lane);
                        if(lane == 0){
                            if( p1_p2_dist < max_neighbor_sqr_dist_sm[p1] ){
                                int pos = atomicAdd(&n_candidates_sm, 1);
                                candidate_dist_sm[pos] = p1_p2_dist;
                                candidate_idx_sm[pos] = real_p2;
                            }
                        }
                    }
                    __syncthreads();
                    // #if DEBUG_BUCKET_EXPLORING
                    //     if(lane == 0)
                    //         printf("DEBUG_BUCKET_EXPLORING in %s %d: Distancance calculation for point %d has finished...\n", __FILE__,__LINE__, real_p1 );
                    // #endif
                    int knn_id = sm_point_id[p1]*K;
                    for(int c=0; c < n_candidates_sm; c++){
                        #if DEBUG_BUCKET_EXPLORING
                            if(c < 0 || c >= N){
                                if(threadIdx.x == 0)
                                    printf("ERROR in %s %d: candidate index %d out of bounds for candidate arrays of size %d, n_candidates_sm=%d\n", __FILE__,__LINE__, c, N, n_candidates_sm);
                                break;
                            }
                            // else {
                            //     if(threadIdx.x == 0 && blockIdx.x == 3)
                            //         printf("DEBUG_BUCKET_EXPLORING in %s %d: Evaluating candidate %d for point %d with distance %f\n", __FILE__,__LINE__, c, real_p1, SQRT_FUNC(candidate_dist_sm[c]) );
                            // }
                        #endif
                        RSFK_typepoints c_dist = candidate_dist_sm[c];
                        if(c_dist < max_neighbor_sqr_dist_sm[p1]){
                            // update knn graph
                            if(threadIdx.x == 0){
                                int c_idx = candidate_idx_sm[c];
                                int old_neighbor = max_neighbor_pos[p1];
                                #if DEBUG_BUCKET_EXPLORING
                                    if(old_neighbor < 0 || old_neighbor >= N*K){
                                        printf("ERROR in %s %d: old_neighbor index %d out of bounds for knn_indices array of size %d*%d=%d\n", __FILE__,__LINE__, old_neighbor, N, K, N*K);
                                    } 
                                    // else {
                                    //     if(threadIdx.x == 0 && blockIdx.x == 3)
                                    //         printf("DEBUG_BUCKET_EXPLORING in %s %d: Updating knn for point %d: replacing neighbor %d (dist %f) with neighbor %d (dist %f)\n", 
                                    //             __FILE__,__LINE__, 
                                    //             real_p1, 
                                    //             knn_indices[old_neighbor], 
                                    //             SQRT_FUNC(knn_sqr_dist[old_neighbor]),
                                    //             c_idx,
                                    //             SQRT_FUNC(c_dist)
                                    //         );
                                    // }
                                    // printf("DEBUG_BUCKET_EXPLORING in %s %d: Updating knn for point %d: replacing neighbor %d (dist %f) with neighbor %d (dist %f)\n", 
                                    //     __FILE__,__LINE__, 
                                    //     real_p1, 
                                    //     knn_indices[old_neighbor], 
                                    //     SQRT_FUNC(knn_sqr_dist[old_neighbor]),
                                    //     c_idx,
                                    //     SQRT_FUNC(c_dist)
                                    // );
                                #endif
                                knn_sqr_dist[old_neighbor] = c_dist;
                                knn_indices[old_neighbor] = c_idx;
                                max_neighbor_sqr_dist_sm[p1] = -1.0f;
                            }
                            __syncthreads();

                            // recompute max neighbor for p1
                            RSFK_typepoints local_max_position = -1;
                            RSFK_typepoints local_max_dist = -1.0f;
                            if(wid == 0){
                                float local_max_dist;
                                int local_max_position;

                                warp_find_max(&knn_sqr_dist[knn_id], K, lane, local_max_dist, local_max_position);

                                if(lane == 0){
                                    max_neighbor_sqr_dist_sm[p1] = local_max_dist;
                                    max_neighbor_pos[p1] = local_max_position + knn_id;
                                }

                            }
                            __syncthreads();
                        }
                    }


                    if(threadIdx.x == 0){
                        point_to_explore_sm = -1;
                        n_candidates_sm = 0;
                    }
                    __syncthreads();
                    if(lane == 0)
                        atomicMax(&point_to_explore_sm, pte);
                    __syncthreads();
                }
            }

            #if DEBUG_BUCKET_EXPLORING
                if(threadIdx.x == 0)
                    printf("DEBUG_BUCKET_EXPLORING in %s %d: Finished exploring bucket %d for bucket %d\n", __FILE__,__LINE__, pb, blockIdx.x );
            #endif

            __syncthreads();
            if(threadIdx.x == 0){
                max_neigh_dist_bucket_sm = -1.0f;
                bucket_to_explore_sm = -1;
            }
            __syncthreads();
            // update max_neigh_dist_bucket_sm
            if(wid == 0){
                float local_max = warp_find_max(max_neighbor_sqr_dist_sm, bsize, lane);
                if(lane == 0)
                    max_neigh_dist_bucket_sm = SQRT_FUNC(local_max);
            }
            if(lane == 0){
                atomicMax(&bucket_to_explore_sm, bte);
            }
            __syncthreads();


            // #if DEBUG_BUCKET_EXPLORING
            //     if(threadIdx.x == 0)
            //         printf("DEBUG_BUCKET_EXPLORING in %s %d: Next bucket to explore for bucket %d is %d\n", __FILE__,__LINE__, blockIdx.x, bucket_to_explore_sm );
            // #endif
        }

    }
    #if COUNT_FILTER_EFFECTIVENESS
        if(lane == 0){
        //     atomicAdd(&global_n_filters_skips[0], n_filter1_skips);
        //     atomicAdd(&global_n_filters_calc_avoided[0], n_filter1_calc_avoided);
        //     atomicAdd(&global_n_filters_skips[1], n_filter2_skips);
        //     atomicAdd(&global_n_filters_calc_avoided[1], n_filter2_calc_avoided);
        //     atomicAdd(&global_n_filters_skips[2], n_filter3_skips);
        //     atomicAdd(&global_n_filters_calc_avoided[2], n_filter3_calc_avoided);
        //     atomicAdd(&global_n_filters_skips[3], n_filter4_skips);
        //     atomicAdd(&global_n_filters_calc_avoided[3], n_filter4_calc_avoided);

            for(int i=0; i<4; i++){
                atomicAdd(&global_total_test[i], total_filter_tests[i]);
            }
        }
    #endif

    #if DEBUG_BUCKET_EXPLORING
        if(threadIdx.x == 0)
            printf("DEBUG_BUCKET_EXPLORING in %s %d: Finished exploring bucket %d\n", __FILE__,__LINE__, blockIdx.x );
    #endif

}

void bucket_exploring(
    float* device_points,
    thrust::device_vector<int> &device_knn_indices,
    thrust::device_vector<RSFK_typepoints> &device_knn_sqr_distances,
    uint *d_kmeans_labels,
    float *d_kmeans_centroids,
    float *d_distances_to_centroids,
    int K, int N, int D, int VERBOSE, TreeInfo tinfo
){

	// ALLOC MEMORY
	cudaError_t err = cudaSuccess;
	int devUsed = 0;
	cudaSetDevice(devUsed);
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, devUsed);

	int max_threads = deviceProp.maxThreadsPerBlock;
	if(max_threads > MAX_THREADS && VERBOSE){
		printf("WA in %s %d: The macro MAX_THREADS (%d) is lower than the device max threads per block (%d). \n", __FILE__,__LINE__, MAX_THREADS,max_threads);
		max_threads = MAX_THREADS;
	}
	int nthreads = deviceProp.maxThreadsPerMultiProcessor / 2;
	if(nthreads > deviceProp.maxThreadsPerBlock) nthreads = deviceProp.maxThreadsPerBlock;
	int nblocks = deviceProp.multiProcessorCount*(deviceProp.maxThreadsPerMultiProcessor/nthreads);
	int nwarps = nthreads / WARP_SIZE;

    int total_buckets = tinfo.total_leaves;
    int max_bucket_size = tinfo.max_child;
    thrust::device_vector<int> device_nodes_buckets = tinfo.device_nodes_buckets;
    thrust::device_vector<int> device_bucket_sizes = tinfo.device_bucket_sizes;

    // Print parameters
    if(VERBOSE >= 2){
        printf("Calling bucket_exploring with parameters:\n",
                total_buckets, nthreads);
        printf("KNN from buckets parameters:\n");
        printf("  N: %d\n", N);
        printf("  D: %d\n", D);
        printf("  K: %d\n", K);
        printf("  total_buckets: %d\n", total_buckets);
        printf("  max_bucket_size: %d\n", max_bucket_size);
    }

    // Before calling, ensure that max_bucket_size is < 1024
    if(max_bucket_size > 1024){
        printf("%s:%d: Error: max_bucket_size (%d) is greater than 1024,"
               " which is not supported by the current KNN bucket"
               " computation kernel implementation.\n",
               __FILE__, __LINE__, max_bucket_size);
        exit(1);
    }

    GpuPtr<float> d_bucket_sqrd_radius(total_buckets);
    uint shared_mem_size = total_buckets * sizeof(float);
    if(shared_mem_size > deviceProp.sharedMemPerBlock){
            #if DEBUG_PYTHON
            fprintf(stderr,"Shared memory per block (%d bytes) is less than required (%d bytes). Using non-persistent kernel.\n",
            #endif
                deviceProp.sharedMemPerBlock, shared_mem_size);
        compute_bucket_radii_not_persistent<<<total_buckets,nthreads,shared_mem_size>>>(
            d_distances_to_centroids,
            d_kmeans_labels,
            d_bucket_sqrd_radius.ptr(),
            N, total_buckets);
    } else {
            #if DEBUG_PYTHON
            fprintf(stderr,"Using persistent kernel for computing bucket squared radii.\n");
            #endif            
        d_bucket_sqrd_radius.zero();
        compute_bucket_radii_persistent<<<nblocks,nthreads,shared_mem_size>>>(
            d_distances_to_centroids,
            d_kmeans_labels,
            d_bucket_sqrd_radius.ptr(),
            N, total_buckets);
    }
    cudaDeviceSynchronize();
    gpuErrchk( cudaPeekAtLastError() );


    #if COUNT_FILTER_EFFECTIVENESS
        GpuPtr<int> global_n_filters_skips(4);
        GpuPtr<int> global_n_filters_calc_avoided(4);
        GpuPtr<int> global_total_test(4);

        global_n_filters_skips.zero();
        global_n_filters_calc_avoided.zero();
        global_total_test.zero();
    #endif


    bucket_exploring_kernel<<<total_buckets, nthreads>>>(
        device_points,
        thrust::raw_pointer_cast(device_knn_indices.data()),
        thrust::raw_pointer_cast(device_knn_sqr_distances.data()),
        d_kmeans_labels,
        d_kmeans_centroids,
        d_distances_to_centroids,
        d_bucket_sqrd_radius.ptr(),
        thrust::raw_pointer_cast(device_nodes_buckets.data()),
        thrust::raw_pointer_cast(device_bucket_sizes.data()),
        total_buckets,
        max_bucket_size,
        K, N, D
        #if COUNT_FILTER_EFFECTIVENESS
        , global_n_filters_skips.ptr(),
        global_n_filters_calc_avoided.ptr(),
        global_total_test.ptr()
        #endif
    );

    cudaDeviceSynchronize();
    gpuErrchk( cudaPeekAtLastError() );

    #if COUNT_FILTER_EFFECTIVENESS
        thrust::host_vector<int> h_global_n_filters_skips(4);
        thrust::host_vector<int> h_global_n_filters_calc_avoided(4);
        thrust::host_vector<int> h_global_total_test(4);

        cudaMemcpy(thrust::raw_pointer_cast(h_global_n_filters_skips.data()), global_n_filters_skips.ptr(), sizeof(int)*4, cudaMemcpyDeviceToHost);
        cudaMemcpy(thrust::raw_pointer_cast(h_global_n_filters_calc_avoided.data()), global_n_filters_calc_avoided.ptr(), sizeof(int)*4, cudaMemcpyDeviceToHost);
        cudaMemcpy(thrust::raw_pointer_cast(h_global_total_test.data()), global_total_test.ptr(), sizeof(int)*4, cudaMemcpyDeviceToHost);

        // if(VERBOSE){
            for(int i=0; i<4; i++){
                fprintf(stderr, "Filter %d: total tests = %d, skips = %d, calculations avoided = %d\n",
                    i+1,
                    h_global_total_test[i],
                    h_global_n_filters_skips[i],
                    h_global_n_filters_calc_avoided[i]);
            }
        // }
    #endif
}