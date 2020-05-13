#ifndef __COMPUTE_KNN_FROM_BUCKETS__H
#define __COMPUTE_KNN_FROM_BUCKETS__H


#include "../../kernels/compute_knn_from_buckets.cu"

// __device__
// inline
// float euclidean_distance_sqr(typepoints* v1, typepoints* v2, int D);

__device__
inline
float euclidean_distance_sqr(int p1, int p2, typepoints* points, int D, int N);


__device__
inline
float euclidean_distance_sqr_small_block(int p1, int p2, typepoints* local_points,
                                         typepoints* points, int D, int N);



__device__
inline
void euclidean_distance_sqr_coalesced_atomic(int p1, int p2, typepoints* points, int D,
                                             int N, int lane, typepoints* diff_sqd);


#if EUCLIDEAN_DISTANCE_VERSION==EDV_ATOMIC_OK

   __device__
   inline
   void euclidean_distance_sqr_coalesced(int p1, int p2, typepoints* points, int D,
                                         int N, int lane, typepoints* diff_sqd);

#elif EUCLIDEAN_DISTANCE_VERSION==EDV_ATOMIC_CSE  // common subexpression elimination

   __device__
   inline
   void euclidean_distance_sqr_coalesced(int p1, int p2, typepoints* points, int D,
                                         int N, int lane, typepoints* diff_sqd);

#elif EUCLIDEAN_DISTANCE_VERSION==EDV_NOATOMIC

   __device__
   inline
   void euclidean_distance_sqr_coalesced(int p1, int p2, typepoints* points, int D,
                                         int N, int lane, typepoints* diff_sqd);

#elif EUCLIDEAN_DISTANCE_VERSION==EDV_NOATOMIC_NOSHM

   __device__                // NOTE: value returned in register (NO SHM)
   inline                    // function return type CHANGED
   typepoints euclidean_distance_sqr_coalesced(int p1, int p2, typepoints* points, int D,
                                         int N, int lane);

#elif EUCLIDEAN_DISTANCE_VERSION==EDV_WARP_REDUCE_XOR

   __device__
   inline
   void euclidean_distance_sqr_coalesced(int p1, int p2, typepoints* points, int D,
                                         int N, int lane, typepoints* diff_sqd);

#elif EUCLIDEAN_DISTANCE_VERSION==EDV_WARP_REDUCE_XOR_NOSHM

   __device__                // NOTE: value returned in register (NO SHM)
   inline                    // function return type CHANGED
   typepoints euclidean_distance_sqr_coalesced(int p1, int p2, typepoints* points, int D,
                                         int N, int lane);

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
                              int MAX_TREE_CHILD, int total_buckets);

// Assign a bucket (leaf in the tree) to each warp and a point to each thread (persistent kernel)
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
                              int MAX_TREE_CHILD, int total_buckets);



// Assign a bucket (leaf in the tree) to each warp and a point to each thread (persistent kernel)
__global__
void compute_knn_from_buckets_perblock_coalesced_symmetric_dividek(int* points_parent,
                              int* points_depth,
                              int* accumulated_nodes_count,
                              typepoints* points,
                              int* node_idx_to_leaf_idx,
                              int* nodes_bucket,
                              int* bucket_size,
                              int* knn_indices,
                              typepoints* knn_sqr_dist,
                              int N, int D, int max_bucket_size, int K,
                              int MAX_TREE_CHILD, int total_buckets);

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
                              int MAX_TREE_CHILD, int total_buckets);

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
                              int MAX_TREE_CHILD);



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
                              int MAX_TREE_CHILD, int total_buckets);

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
                              int MAX_TREE_CHILD, int total_buckets);

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
                              int MAX_TREE_CHILD);

#endif