#ifndef __COMPUTE_KNN_FROM_BUCKETS__CU
#define __COMPUTE_KNN_FROM_BUCKETS__CU

#include "../include/common.h"

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
void euclidean_distance_sqr_coalesced_atomic(int p1, int p2, typepoints* points, int D,
                                             int N, int lane, typepoints* diff_sqd)
{
    typepoints diff;

    for(int i=lane; i < D; i+=32){
        diff = points[get_point_idx(p1,i,N,D)] - points[get_point_idx(p2,i,N,D)];
        atomicAdd(diff_sqd,diff*diff);
    }
}


#if EUCLIDEAN_DISTANCE_VERSION==EDV_ATOMIC_OK

   __device__
   inline
   void euclidean_distance_sqr_coalesced(int p1, int p2, typepoints* points, int D,
                                         int N, int lane, typepoints* diff_sqd)
   {
       typepoints diff;
       typepoints s = 0.0f;
       for(int i=lane; i < D; i+=32){
           diff = points[get_point_idx(p1,i,N,D)] - points[get_point_idx(p2,i,N,D)];
           s+=diff*diff;
           // atomicAdd(diff_sqd,diff*diff);
       }
       atomicAdd(diff_sqd,s);
   }

#elif EUCLIDEAN_DISTANCE_VERSION==EDV_ATOMIC_CSE  // common subexpression elimination

   __device__
   inline
   void euclidean_distance_sqr_coalesced(int p1, int p2, typepoints* points, int D,
                                         int N, int lane, typepoints* diff_sqd)
   {
       register typepoints diff;
       register typepoints s = 0.0f;
       // #define get_point_idx(point,dimension,N,D) (point*D+dimension)
       register typepoints* _p1 = &points[p1*D];
       register typepoints* _p2 = &points[p2*D];
       for(register int i=lane; i < D; i+=32){
           diff = _p1[i] - _p2[i];
           s+=diff*diff;
           // atomicAdd(diff_sqd,diff*diff);
       }
       atomicAdd(diff_sqd,s);
   }

#elif EUCLIDEAN_DISTANCE_VERSION==EDV_NOATOMIC

   __device__
   inline
   void euclidean_distance_sqr_coalesced(int p1, int p2, typepoints* points, int D,
                                         int N, int lane, typepoints* diff_sqd)
   {
       typepoints diff;
       typepoints s = 0.0f;
       for(int i=lane; i < D; i+=32){
           diff = points[get_point_idx(p1,i,N,D)] - points[get_point_idx(p2,i,N,D)];
           s+=diff*diff;
           // atomicAdd(diff_sqd,diff*diff);
       }
       //atomicAdd(diff_sqd,s);
       // do a simpler shfl_down warp reduce
       s += __shfl_down_sync( 0xffffffff, s, 16 );
       s += __shfl_down_sync( 0xffffffff, s,  8 );
       s += __shfl_down_sync( 0xffffffff, s,  4 );
       s += __shfl_down_sync( 0xffffffff, s,  2 );
       s += __shfl_down_sync( 0xffffffff, s,  1 );
       // lane 0 stores result in SHM
       if( threadIdx.x & 0x1f )
           *diff_sqd = s;
   }

#elif EUCLIDEAN_DISTANCE_VERSION==EDV_NOATOMIC_NOSHM

   __device__                // NOTE: value returned in register (NO SHM)
   inline                    // function return type CHANGED
   typepoints euclidean_distance_sqr_coalesced(int p1, int p2, typepoints* points, int D,
                                         int N, int lane)
   {
       typepoints diff;
       typepoints s = 0.0f;
       
       for(int i=lane; i < D; i+=32){
           diff = points[get_point_idx(p1,i,N,D)] - points[get_point_idx(p2,i,N,D)];
           s+=diff*diff;
           // atomicAdd(diff_sqd,diff*diff);
       }
       //atomicAdd(diff_sqd,s);
       // do a simpler shfl_down warp reduce
       s += __shfl_down_sync( 0xffffffff, s, 16 ); // assuming warpSize=32
       s += __shfl_down_sync( 0xffffffff, s,  8 ); // assuming warpSize=32
       s += __shfl_down_sync( 0xffffffff, s,  4 ); // assuming warpSize=32
       s += __shfl_down_sync( 0xffffffff, s,  2 ); // assuming warpSize=32
       s += __shfl_down_sync( 0xffffffff, s,  1 ); // assuming warpSize=32
       // broadcast reduced value to all threads in warp (so they can return the value)
       __shfl_down_sync( 0xffffffff, s,  1 );
       return s;
   }

#elif EUCLIDEAN_DISTANCE_VERSION==EDV_WARP_REDUCE_XOR

   __device__
   inline
   void euclidean_distance_sqr_coalesced(int p1, int p2, typepoints* points, int D,
                                         int N, int lane, typepoints* diff_sqd)
   {
       typepoints diff;
       typepoints s = 0.0f;
       for(int i=lane; i < D; i+=32){
           diff = points[get_point_idx(p1,i,N,D)] - points[get_point_idx(p2,i,N,D)];
           s+=diff*diff;
           // atomicAdd(diff_sqd,diff*diff);
       }
       //atomicAdd(diff_sqd,s);
       s += __shfl_xor_sync( 0xffffffff, s,  1); // assuming warpSize=32
       s += __shfl_xor_sync( 0xffffffff, s,  2); // assuming warpSize=32
       s += __shfl_xor_sync( 0xffffffff, s,  4); // assuming warpSize=32
       s += __shfl_xor_sync( 0xffffffff, s,  8); // assuming warpSize=32
       s += __shfl_xor_sync( 0xffffffff, s, 16); // assuming warpSize=32
       // lane 0 stores result in SHM
       if( threadIdx.x & 0x1f )
           *diff_sqd = s;
   }

#elif EUCLIDEAN_DISTANCE_VERSION==EDV_WARP_REDUCE_XOR_NOSHM

   __device__                // NOTE: value returned in register (NO SHM)
   inline                    // function return type CHANGED
   typepoints euclidean_distance_sqr_coalesced(int p1, int p2, typepoints* points, int D,
                                         int N, int lane)
   {
       typepoints diff;
       typepoints s = 0.0f;
       
       for(int i=lane; i < D; i+=32){
            diff = points[get_point_idx(p1,i,N,D)] - points[get_point_idx(p2,i,N,D)];
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

#endif


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
    typepoints max_dist_val;
    
    int knn_id;
    int lane = threadIdx.x % 32; // my id on warp
    

    #if EUCLIDEAN_DISTANCE_VERSION!=EDV_NOATOMIC_NOSHM && EUCLIDEAN_DISTANCE_VERSION!=EDV_WARP_REDUCE_XOR_NOSHM
    __shared__ typepoints candidate_dist_val[1024];
    int init_warp_on_block = (threadIdx.x/32)*32;
    #else
    typepoints candidate_dist_val, tmp_dist_val;
    #endif

    int bid, p, _p, i, j;
    int tmp_candidate, tmp_p;
    
    for(bid = tid/32; bid < total_buckets; bid+=blockDim.x*gridDim.x/32){
        current_bucket_size = bucket_size[bid];
        for(_p = lane; __any_sync(__activemask(),_p < current_bucket_size); _p+=32){
            if(_p < current_bucket_size){
                p = nodes_bucket[bid*max_bucket_size + _p];
                knn_id = p*K;
                parent_id = accumulated_nodes_count[points_depth[p]] + points_parent[p];
                current_bucket_size = bucket_size[node_idx_to_leaf_idx[parent_id]];

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
            }
            for(i=0; i < current_bucket_size; ++i){

                candidate_point = -1;
                if(_p < current_bucket_size){
                    candidate_point = nodes_bucket[node_idx_to_leaf_idx[parent_id]*max_bucket_size + i];
                    
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

                #if EUCLIDEAN_DISTANCE_VERSION!=EDV_NOATOMIC_NOSHM && EUCLIDEAN_DISTANCE_VERSION!=EDV_WARP_REDUCE_XOR_NOSHM
                candidate_dist_val[threadIdx.x] = 0.0f;
                #endif

                for(j=0; j < 32; ++j){
                    __syncwarp();
                    tmp_candidate = __shfl_sync(0xffffffff, candidate_point, j);
                    if(tmp_candidate == -1) continue;
                    tmp_p = __shfl_sync(0xffffffff, p, j);
                    #if EUCLIDEAN_DISTANCE_VERSION!=EDV_NOATOMIC_NOSHM && EUCLIDEAN_DISTANCE_VERSION!=EDV_WARP_REDUCE_XOR_NOSHM
                    euclidean_distance_sqr_coalesced(tmp_candidate, tmp_p, points, D, N,
                                                    lane,
                                                    &candidate_dist_val[init_warp_on_block+j]);
                    #else
                    tmp_dist_val = euclidean_distance_sqr_coalesced(tmp_candidate, tmp_p, points, D, N, lane);
                    if(lane == j) candidate_dist_val = tmp_dist_val;
                    #endif
                }
                if(candidate_point == -1) continue;

                // If the candidate is closer than the pre-computed furthest point,
                // switch them
                #if EUCLIDEAN_DISTANCE_VERSION!=EDV_NOATOMIC_NOSHM && EUCLIDEAN_DISTANCE_VERSION!=EDV_WARP_REDUCE_XOR_NOSHM
                if(candidate_dist_val[threadIdx.x] < max_dist_val){
                #else
                if(candidate_dist_val < max_dist_val){
                #endif
                    knn_indices[max_id_point] = candidate_point;
                    #if EUCLIDEAN_DISTANCE_VERSION!=EDV_NOATOMIC_NOSHM && EUCLIDEAN_DISTANCE_VERSION!=EDV_WARP_REDUCE_XOR_NOSHM
                    knn_sqr_dist[max_id_point] = candidate_dist_val[threadIdx.x];
                    #else
                    knn_sqr_dist[max_id_point] = candidate_dist_val;
                    #endif
                    // Also update the furthest point that will be used in the next
                    // comparison
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

// Assign a bucket (leaf in the tree) to each block (persistent kernel)
// In this kernel, the redundant computation of symetric distances is avoided

// Since different points neighborhood may be updated by different threads,
// a lock system must be implemented
__global__
void compute_knn_from_buckets_perblock_coalesced_symmetric(int* points_parent,
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
    int cbs; // cbs = current bucket size
    
    int knn_id;
    int wid = threadIdx.x / 32; // my id on warp
    int lane = threadIdx.x % 32; // my id on warp
    
    #if EUCLIDEAN_DISTANCE_VERSION!=EDV_NOATOMIC_NOSHM && EUCLIDEAN_DISTANCE_VERSION!=EDV_WARP_REDUCE_XOR_NOSHM
    __shared__ typepoints candidate_dist_val[32];
    #else
    typepoints candidate_dist_val;
    #endif

    int bid, p1, p2, real_p1, real_p2, _p, i, j;
    
    __shared__ int sm_leaf_bucket[300];
    __shared__ typepoints max_dist_val[300];
    __shared__ int max_position[300];

    int done_p1, done_p2;
    __shared__ int lock_point[300];
    
    bid = blockIdx.x;
    cbs = bucket_size[bid];
    for(i=threadIdx.x; i < cbs; i+=blockDim.x){
        p1 = nodes_bucket[bid*max_bucket_size + i];
        sm_leaf_bucket[i] = p1;
        lock_point[i] = 0;

        knn_id = p1*K;

        max_position[i] = knn_id;
        max_dist_val[i] = knn_sqr_dist[knn_id];
        // Finds the index of the furthest point from the current result of knn_indices
        // and the distance between them
        for(j=1; j < K; ++j){
            if(knn_sqr_dist[knn_id+j] > max_dist_val[i]){
                max_position[i] = knn_id+j; // The initial point is not necessarily in the bucket
                max_dist_val[i] = knn_sqr_dist[knn_id+j];
            }
        }
    }

    __syncthreads();
    
    for(_p = wid; _p < (cbs*cbs - cbs)/2; _p+=blockDim.x/32){
        p1 = cbs - 2 - floor(sqrt((float)((-8*_p + 4*cbs*(cbs-1)-7)))/2.0 - 0.5);
        p2 = _p + p1 + 1 - cbs*(cbs-1)/2 + (cbs-p1)*((cbs-p1)-1)/2;
        real_p1 = sm_leaf_bucket[p1];
        real_p2 = sm_leaf_bucket[p2];

        #if EUCLIDEAN_DISTANCE_VERSION!=EDV_NOATOMIC_NOSHM && EUCLIDEAN_DISTANCE_VERSION!=EDV_WARP_REDUCE_XOR_NOSHM
        candidate_dist_val[wid] = 0.0f;
        #endif

        __syncwarp();
        #if EUCLIDEAN_DISTANCE_VERSION!=EDV_NOATOMIC_NOSHM && EUCLIDEAN_DISTANCE_VERSION!=EDV_WARP_REDUCE_XOR_NOSHM
        euclidean_distance_sqr_coalesced(real_p1,
                                         real_p2,
                                         points, D, N,
                                         lane,
                                         &candidate_dist_val[wid]);
        #else
        candidate_dist_val = euclidean_distance_sqr_coalesced(real_p1,
                                                              real_p2,
                                                              points, D, N, lane);
        #endif
        __syncwarp();
        
        if(lane == 0){
            #if EUCLIDEAN_DISTANCE_VERSION!=EDV_NOATOMIC_NOSHM && EUCLIDEAN_DISTANCE_VERSION!=EDV_WARP_REDUCE_XOR_NOSHM
            done_p1 = candidate_dist_val[wid] >= max_dist_val[p1];
            done_p2 = candidate_dist_val[wid] >= max_dist_val[p2];
            #else
            done_p1 = candidate_dist_val >= max_dist_val[p1];
            done_p2 = candidate_dist_val >= max_dist_val[p2];
            #endif

            for(j=0; j < K && (!done_p1 || !done_p2); ++j){
                done_p1 |= real_p2 == knn_indices[real_p1*K+j];
                done_p2 |= real_p1 == knn_indices[real_p2*K+j];
            }

            while(!done_p1 || !done_p2){
                if(!done_p1 && !atomicCAS(&lock_point[p1], 0, 1)){
                    done_p1 = 1;
                    // If the candidate is closer than the pre-computed furthest point,
                    // switch them
                    #if EUCLIDEAN_DISTANCE_VERSION!=EDV_NOATOMIC_NOSHM && EUCLIDEAN_DISTANCE_VERSION!=EDV_WARP_REDUCE_XOR_NOSHM
                    if(candidate_dist_val[wid] < max_dist_val[p1]){
                    #else
                    if(candidate_dist_val < max_dist_val[p1]){
                    #endif
                        knn_indices[max_position[p1]] = real_p2;
                        #if EUCLIDEAN_DISTANCE_VERSION!=EDV_NOATOMIC_NOSHM && EUCLIDEAN_DISTANCE_VERSION!=EDV_WARP_REDUCE_XOR_NOSHM
                        knn_sqr_dist[max_position[p1]] = candidate_dist_val[wid];
                        #else
                        knn_sqr_dist[max_position[p1]] = candidate_dist_val;
                        #endif

                        // Also update the furthest point that will be used in the next
                        // comparison
                        knn_id = real_p1*K;
                        max_position[p1] = knn_id;
                        max_dist_val[p1] = knn_sqr_dist[knn_id];
                        for(j=1; j < K; ++j){
                            if(knn_sqr_dist[knn_id+j] > max_dist_val[p1]){
                                max_position[p1] = knn_id+j;
                                max_dist_val[p1] = knn_sqr_dist[knn_id+j];
                            }
                        }
                    }
                    atomicExch(&lock_point[p1], 0);
                }

                if(!done_p2 && !atomicCAS(&lock_point[p2], 0, 1)){
                    done_p2 = 1;
                    // If the candidate is closer than the pre-computed furthest point,
                    // switch them
                    #if EUCLIDEAN_DISTANCE_VERSION!=EDV_NOATOMIC_NOSHM && EUCLIDEAN_DISTANCE_VERSION!=EDV_WARP_REDUCE_XOR_NOSHM
                    if(candidate_dist_val[wid] < max_dist_val[p2]){
                    #else
                    if(candidate_dist_val < max_dist_val[p2]){
                    #endif
                        knn_indices[max_position[p2]] = real_p1;
                        #if EUCLIDEAN_DISTANCE_VERSION!=EDV_NOATOMIC_NOSHM && EUCLIDEAN_DISTANCE_VERSION!=EDV_WARP_REDUCE_XOR_NOSHM
                        knn_sqr_dist[max_position[p2]] = candidate_dist_val[wid];
                        #else
                        knn_sqr_dist[max_position[p2]] = candidate_dist_val;
                        #endif
        
                        // Also update the furthest point that will be used in the next
                        // comparison
                        knn_id = real_p2*K;
                        max_position[p2] = knn_id;
                        max_dist_val[p2] = knn_sqr_dist[knn_id];
                        for(j=1; j < K; ++j){
                            if(knn_sqr_dist[knn_id+j] > max_dist_val[p2]){
                                max_position[p2] = knn_id+j;
                                max_dist_val[p2] = knn_sqr_dist[knn_id+j];
                            }
                        }
                    }
                    atomicExch(&lock_point[p2], 0);
                }
            }
        }
        __syncwarp();
    }
    
}


// This kernel is a optmization of compute_knn_from_buckets_perblock_coalesced_symmetric kernel
// The optimization consists use and communicate idle threads during lock system
__global__
void compute_knn_from_buckets_perblock_coalesced_symmetric_dividek(
                              typepoints* points,
                              int* nodes_bucket,
                              int* bucket_size,
                              int* knn_indices,
                              typepoints* knn_sqr_dist,
                              int N, int D, int max_bucket_size, int K,
                              int MAX_TREE_CHILD, int total_buckets)
{
    int cbs; // cbs = current bucket size
    
    int knn_id;
    int wid = threadIdx.x / 32; // my warp id
    int lane = threadIdx.x % 32; // my id on warp
    

    #if EUCLIDEAN_DISTANCE_VERSION!=EDV_NOATOMIC_NOSHM && EUCLIDEAN_DISTANCE_VERSION!=EDV_WARP_REDUCE_XOR_NOSHM
    __shared__ typepoints candidate_dist_val[32];
    #else
    typepoints candidate_dist_val;
    #endif

    int bid, p1, p2, real_p1, real_p2, _p, i, j;
    
    __shared__ int sm_leaf_bucket[1024];
    __shared__ typepoints max_dist_val[1024];
    __shared__ int max_position[1024];
    
    int local_max_position, tmp_max_position;
    typepoints local_max_dist, tmp_max_dist;

    int done_p1, done_p2;
    __shared__ int lock_point[1024];

    // Non persistent kernel seems to be more efficient
    bid=blockIdx.x;
    __syncthreads();
    cbs = bucket_size[bid];
    for(i=threadIdx.x; i < cbs; i+=blockDim.x){
        p1 = nodes_bucket[bid*max_bucket_size + i];
        sm_leaf_bucket[i] = p1;
        lock_point[i] = 0;

        knn_id = p1*K;

        max_position[i] = knn_id;
        max_dist_val[i] = knn_sqr_dist[knn_id];
        // Finds the index of the furthest point from the current result of knn_indices
        // and the distance between them
        for(j=1; j < K; ++j){
            if(knn_sqr_dist[knn_id+j] > max_dist_val[i]){
                max_position[i] = knn_id+j; // The initial point is not necessarily in the bucket
                max_dist_val[i] = knn_sqr_dist[knn_id+j];
            }
        }

    }

    __syncthreads();
    
    for(_p = wid; _p < (cbs*cbs - cbs)/2; _p+=blockDim.x/32){
        p1 = cbs - 2 - floor(sqrt((float)((-8*_p + 4*cbs*(cbs-1)-7)))/2.0 - 0.5);
        p2 = _p + p1 + 1 - cbs*(cbs-1)/2 + (cbs-p1)*((cbs-p1)-1)/2;
        real_p1 = sm_leaf_bucket[p1];
        real_p2 = sm_leaf_bucket[p2];
        // __syncwarp();


        #if EUCLIDEAN_DISTANCE_VERSION!=EDV_NOATOMIC_NOSHM && EUCLIDEAN_DISTANCE_VERSION!=EDV_WARP_REDUCE_XOR_NOSHM
        candidate_dist_val[wid] = 0.0f;
        #endif

        // __syncwarp();
        // __syncthreads();
        // __syncwarp();
        #if EUCLIDEAN_DISTANCE_VERSION!=EDV_NOATOMIC_NOSHM && EUCLIDEAN_DISTANCE_VERSION!=EDV_WARP_REDUCE_XOR_NOSHM
        euclidean_distance_sqr_coalesced(real_p1,
                                        real_p2,
                                        points, D, N,
                                        lane,
                                        &candidate_dist_val[wid]);
        __syncwarp();
        #else
        candidate_dist_val = euclidean_distance_sqr_coalesced(real_p1,
                                                            real_p2,
                                                            points, D, N, lane);
        #endif
        
        #if EUCLIDEAN_DISTANCE_VERSION!=EDV_NOATOMIC_NOSHM && EUCLIDEAN_DISTANCE_VERSION!=EDV_WARP_REDUCE_XOR_NOSHM
        done_p1 = candidate_dist_val[wid] >= max_dist_val[p1];
        done_p2 = candidate_dist_val[wid] >= max_dist_val[p2];
        #else
        done_p1 = candidate_dist_val >= max_dist_val[p1];
        done_p2 = candidate_dist_val >= max_dist_val[p2];
        #endif
        
        // Verify if the candidate point already is in the knn_indices
        for(j=lane; j < K && (!done_p1 || !done_p2); j+=32){
            done_p1 |= real_p2 == knn_indices[real_p1*K+j];
            done_p2 |= real_p1 == knn_indices[real_p2*K+j];
        }
        
        done_p1 |= __shfl_xor_sync( 0xffffffff, done_p1,  1); // assuming warpSize=32
        done_p1 |= __shfl_xor_sync( 0xffffffff, done_p1,  2); // assuming warpSize=32
        done_p1 |= __shfl_xor_sync( 0xffffffff, done_p1,  4); // assuming warpSize=32
        done_p1 |= __shfl_xor_sync( 0xffffffff, done_p1,  8); // assuming warpSize=32
        done_p1 |= __shfl_xor_sync( 0xffffffff, done_p1, 16); // assuming warpSize=32
        
        done_p2 |= __shfl_xor_sync( 0xffffffff, done_p2,  1); // assuming warpSize=32
        done_p2 |= __shfl_xor_sync( 0xffffffff, done_p2,  2); // assuming warpSize=32
        done_p2 |= __shfl_xor_sync( 0xffffffff, done_p2,  4); // assuming warpSize=32
        done_p2 |= __shfl_xor_sync( 0xffffffff, done_p2,  8); // assuming warpSize=32
        done_p2 |= __shfl_xor_sync( 0xffffffff, done_p2, 16); // assuming warpSize=32

        
        
        while(!done_p1 || !done_p2){
            // if(!done_p1 && !atomicOr(&lock_point[p1], 1)){
            if(!done_p1 && __any_sync(0xffffffff,!atomicCAS(&lock_point[p1], 0, 1))){
                done_p1 = 1;
                // If the candidate is closer than the pre-computed furthest point,
                // switch them
                #if EUCLIDEAN_DISTANCE_VERSION!=EDV_NOATOMIC_NOSHM && EUCLIDEAN_DISTANCE_VERSION!=EDV_WARP_REDUCE_XOR_NOSHM
                if(candidate_dist_val[wid] < max_dist_val[p1]){
                #else
                if(candidate_dist_val < max_dist_val[p1]){
                #endif
                    if(lane == 0){
                        knn_indices[max_position[p1]] = real_p2;
                        #if EUCLIDEAN_DISTANCE_VERSION!=EDV_NOATOMIC_NOSHM && EUCLIDEAN_DISTANCE_VERSION!=EDV_WARP_REDUCE_XOR_NOSHM
                        knn_sqr_dist[max_position[p1]] = candidate_dist_val[wid];
                        #else
                        knn_sqr_dist[max_position[p1]] = candidate_dist_val;
                        #endif
                    }

                    // Also update the furthest point that will be used in the next
                    // comparison
                    knn_id = real_p1*K;

                    local_max_position = -1;
                    local_max_dist = -1.0f;
                    for(j=lane; j < K; j+=32){
                        if(knn_sqr_dist[knn_id+j] > local_max_dist){
                            local_max_position = knn_id+j;
                            local_max_dist = knn_sqr_dist[knn_id+j];
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

                    if(lane == 0){
                        max_dist_val[p1] = local_max_dist;
                        max_position[p1] = local_max_position;
                    }
                }
                
                if(lane == 0) atomicExch(&lock_point[p1], 0);
                // lock_point[p1] = 0;
            }

            // if(!done_p2 && !atomicOr(&lock_point[p2], 1)){
            if(!done_p2 && __any_sync(0xffffffff,!atomicCAS(&lock_point[p2], 0, 1))){
                done_p2 = 1;
                // If the candidate is closer than the pre-computed furthest point,
                // switch them
                #if EUCLIDEAN_DISTANCE_VERSION!=EDV_NOATOMIC_NOSHM && EUCLIDEAN_DISTANCE_VERSION!=EDV_WARP_REDUCE_XOR_NOSHM
                if(candidate_dist_val[wid] < max_dist_val[p2]){
                #else
                if(candidate_dist_val < max_dist_val[p2]){
                #endif
                    if(lane == 0){
                        knn_indices[max_position[p2]] = real_p1;
                        #if EUCLIDEAN_DISTANCE_VERSION!=EDV_NOATOMIC_NOSHM && EUCLIDEAN_DISTANCE_VERSION!=EDV_WARP_REDUCE_XOR_NOSHM
                        knn_sqr_dist[max_position[p2]] = candidate_dist_val[wid];
                        #else
                        knn_sqr_dist[max_position[p2]] = candidate_dist_val;
                        #endif
                    }

                    // Also update the furthest point that will be used in the next
                    // comparison
                    knn_id = real_p2*K;

                    local_max_position = -1;
                    local_max_dist = -1.0f;
                    for(j=lane; j < K; j+=32){
                        if(knn_sqr_dist[knn_id+j] > local_max_dist){
                            local_max_position = knn_id+j;
                            local_max_dist = knn_sqr_dist[knn_id+j];
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

                    if(lane == 0){
                        max_dist_val[p2] = local_max_dist;
                        max_position[p2] = local_max_position;
                    }
                }
                if(lane == 0) atomicExch(&lock_point[p2], 0);
                // lock_point[p2] = 0;
            }
        } //end if(done p2)
        // __syncwarp();
    } // end while(not done)
}

// Assign a bucket (leaf in the tree) to each warp and a point to each thread (persistent kernel)
__global__
void compute_knn_from_buckets_pertile_coalesced_symmetric(int* points_parent,
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
    int cbs; // cbs = current bucket size
    
    int knn_id;
    int wid = threadIdx.x / 32; // my id on warp
    int lane = threadIdx.x % 32; // my id on warp
    
    // extern __shared__ typepoints local_candidate_dist_val[];

    #if EUCLIDEAN_DISTANCE_VERSION!=EDV_NOATOMIC_NOSHM && EUCLIDEAN_DISTANCE_VERSION!=EDV_WARP_REDUCE_XOR_NOSHM
    __shared__ typepoints candidate_dist_val[32];
    #else
    typepoints candidate_dist_val;
    #endif

    int bid, p1, p2, real_p1, real_p2, _p, i, j;
    
    __shared__ int sm_leaf_bucket[300];
    __shared__ typepoints max_dist_val[300];
    __shared__ int max_position[300];

    int done_p1, done_p2;
    __shared__ int lock_point[300];
    
    bid = blockIdx.x;
    cbs = bucket_size[bid];
    for(i=threadIdx.x; i < cbs; i+=blockDim.x){
        p1 = nodes_bucket[bid*max_bucket_size + i];
        sm_leaf_bucket[i] = p1;
        lock_point[i] = 0;

        knn_id = p1*K;

        max_position[i] = knn_id;
        max_dist_val[i] = knn_sqr_dist[knn_id];
        // Finds the index of the furthest point from the current result of knn_indices
        // and the distance between them
        for(j=1; j < K; ++j){
            if(knn_sqr_dist[knn_id+j] > max_dist_val[i]){
                max_position[i] = knn_id+j; // The initial point is not necessarily in the bucket
                max_dist_val[i] = knn_sqr_dist[knn_id+j];
            }
        }

    }

    __syncthreads();
    
    for(_p = wid; _p < (cbs*cbs - cbs)/2; _p+=blockDim.x/32){
        p1 = cbs - 2 - floor(sqrt((float)((-8*_p + 4*cbs*(cbs-1)-7)))/2.0 - 0.5);
        p2 = _p + p1 + 1 - cbs*(cbs-1)/2 + (cbs-p1)*((cbs-p1)-1)/2;
        real_p1 = sm_leaf_bucket[p1];
        real_p2 = sm_leaf_bucket[p2];
        // __syncwarp();


        #if EUCLIDEAN_DISTANCE_VERSION!=EDV_NOATOMIC_NOSHM && EUCLIDEAN_DISTANCE_VERSION!=EDV_WARP_REDUCE_XOR_NOSHM
        candidate_dist_val[wid] = 0.0f;
        #endif

        // __syncwarp();
        // __syncthreads();
        __syncwarp();
        #if EUCLIDEAN_DISTANCE_VERSION!=EDV_NOATOMIC_NOSHM && EUCLIDEAN_DISTANCE_VERSION!=EDV_WARP_REDUCE_XOR_NOSHM
        euclidean_distance_sqr_coalesced(real_p1,
                                         real_p2,
                                         points, D, N,
                                         lane,
                                         &candidate_dist_val[wid]);
        #else
        candidate_dist_val = euclidean_distance_sqr_coalesced(real_p1,
                                                              real_p2,
                                                              points, D, N, lane);
        #endif
        __syncwarp();
        
        if(lane == 0){
            #if EUCLIDEAN_DISTANCE_VERSION!=EDV_NOATOMIC_NOSHM && EUCLIDEAN_DISTANCE_VERSION!=EDV_WARP_REDUCE_XOR_NOSHM
            done_p1 = candidate_dist_val[wid] >= max_dist_val[p1];
            done_p2 = candidate_dist_val[wid] >= max_dist_val[p2];
            #else
            done_p1 = candidate_dist_val >= max_dist_val[p1];
            done_p2 = candidate_dist_val >= max_dist_val[p2];
            #endif

            for(j=0; j < K && (!done_p1 || !done_p2); ++j){
                done_p1 |= real_p2 == knn_indices[real_p1*K+j];
                done_p2 |= real_p1 == knn_indices[real_p2*K+j];
            }

            while(!done_p1 || !done_p2){
                // if(!done_p1 && !atomicOr(&lock_point[p1], 1)){
                if(!done_p1 && !atomicCAS(&lock_point[p1], 0, 1)){
                    done_p1 = 1;
                    // If the candidate is closer than the pre-computed furthest point,
                    // switch them
                    #if EUCLIDEAN_DISTANCE_VERSION!=EDV_NOATOMIC_NOSHM && EUCLIDEAN_DISTANCE_VERSION!=EDV_WARP_REDUCE_XOR_NOSHM
                    if(candidate_dist_val[wid] < max_dist_val[p1]){
                    #else
                    if(candidate_dist_val < max_dist_val[p1]){
                    #endif
                        // if(real_p1 == 500) printf("%d %d %d %f %f %d\n",max_position[p1], max_position[p1]/K, max_position[p1] %K, max_dist_val[p1], candidate_dist_val[wid], real_p2);

                        knn_indices[max_position[p1]] = real_p2;
                        #if EUCLIDEAN_DISTANCE_VERSION!=EDV_NOATOMIC_NOSHM && EUCLIDEAN_DISTANCE_VERSION!=EDV_WARP_REDUCE_XOR_NOSHM
                        knn_sqr_dist[max_position[p1]] = candidate_dist_val[wid];
                        #else
                        knn_sqr_dist[max_position[p1]] = candidate_dist_val;
                        #endif

                        // Also update the furthest point that will be used in the next
                        // comparison
                        knn_id = real_p1*K;
                        max_position[p1] = knn_id;
                        max_dist_val[p1] = knn_sqr_dist[knn_id];
                        for(j=1; j < K; ++j){
                            if(knn_sqr_dist[knn_id+j] > max_dist_val[p1]){
                                max_position[p1] = knn_id+j;
                                max_dist_val[p1] = knn_sqr_dist[knn_id+j];
                            }
                        }
                    }
                    atomicExch(&lock_point[p1], 0);
                    // lock_point[p1] = 0;
                }

                // if(!done_p2 && !atomicOr(&lock_point[p2], 1)){
                if(!done_p2 && !atomicCAS(&lock_point[p2], 0, 1)){
                    done_p2 = 1;
                    // If the candidate is closer than the pre-computed furthest point,
                    // switch them
                    #if EUCLIDEAN_DISTANCE_VERSION!=EDV_NOATOMIC_NOSHM && EUCLIDEAN_DISTANCE_VERSION!=EDV_WARP_REDUCE_XOR_NOSHM
                    if(candidate_dist_val[wid] < max_dist_val[p2]){
                    #else
                    if(candidate_dist_val < max_dist_val[p2]){
                    #endif
                        knn_indices[max_position[p2]] = real_p1;
                        #if EUCLIDEAN_DISTANCE_VERSION!=EDV_NOATOMIC_NOSHM && EUCLIDEAN_DISTANCE_VERSION!=EDV_WARP_REDUCE_XOR_NOSHM
                        knn_sqr_dist[max_position[p2]] = candidate_dist_val[wid];
                        #else
                        knn_sqr_dist[max_position[p2]] = candidate_dist_val;
                        #endif
        
                        // Also update the furthest point that will be used in the next
                        // comparison
                        knn_id = real_p2*K;
                        max_position[p2] = knn_id;
                        max_dist_val[p2] = knn_sqr_dist[knn_id];
                        for(j=1; j < K; ++j){
                            if(knn_sqr_dist[knn_id+j] > max_dist_val[p2]){
                                max_position[p2] = knn_id+j;
                                max_dist_val[p2] = knn_sqr_dist[knn_id+j];
                            }
                        }
                    }
                    atomicExch(&lock_point[p2], 0);
                    // lock_point[p2] = 0;
                }
            }
        }
        __syncwarp();
    }
    
}


#endif