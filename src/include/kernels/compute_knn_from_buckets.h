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

#ifndef __COMPUTE_KNN_FROM_BUCKETS__H
#define __COMPUTE_KNN_FROM_BUCKETS__H


#include "../../kernels/compute_knn_from_buckets.cu"

// __device__
// inline
// float euclidean_distance_sqr(RSFK_typepoints* v1, RSFK_typepoints* v2, int D);

__device__
inline
float euclidean_distance_sqr(int p1, int p2, RSFK_typepoints* points, int D, int N);


__device__
inline
float euclidean_distance_sqr_small_block(int p1, int p2, RSFK_typepoints* local_points,
                                         RSFK_typepoints* points, int D, int N);



__device__
inline
void euclidean_distance_sqr_coalesced_atomic(int p1, int p2, RSFK_typepoints* points, int D,
                                             int N, int lane, RSFK_typepoints* diff_sqd);


#if EUCLIDEAN_DISTANCE_VERSION==EDV_ATOMIC_OK

   __device__
   inline
   void euclidean_distance_sqr_coalesced(int p1, int p2, RSFK_typepoints* points, int D,
                                         int N, int lane, RSFK_typepoints* diff_sqd);

#elif EUCLIDEAN_DISTANCE_VERSION==EDV_ATOMIC_CSE  // common subexpression elimination

   __device__
   inline
   void euclidean_distance_sqr_coalesced(int p1, int p2, RSFK_typepoints* points, int D,
                                         int N, int lane, RSFK_typepoints* diff_sqd);

#elif EUCLIDEAN_DISTANCE_VERSION==EDV_NOATOMIC

   __device__
   inline
   void euclidean_distance_sqr_coalesced(int p1, int p2, RSFK_typepoints* points, int D,
                                         int N, int lane, RSFK_typepoints* diff_sqd);

#elif EUCLIDEAN_DISTANCE_VERSION==EDV_NOATOMIC_NOSHM

   __device__                // NOTE: value returned in register (NO SHM)
   inline                    // function return type CHANGED
   RSFK_typepoints euclidean_distance_sqr_coalesced(int p1, int p2, RSFK_typepoints* points, int D,
                                         int N, int lane);

#elif EUCLIDEAN_DISTANCE_VERSION==EDV_WARP_REDUCE_XOR

   __device__
   inline
   void euclidean_distance_sqr_coalesced(int p1, int p2, RSFK_typepoints* points, int D,
                                         int N, int lane, RSFK_typepoints* diff_sqd);

#elif EUCLIDEAN_DISTANCE_VERSION==EDV_WARP_REDUCE_XOR_NOSHM

   __device__                // NOTE: value returned in register (NO SHM)
   inline                    // function return type CHANGED
   RSFK_typepoints euclidean_distance_sqr_coalesced(int p1, int p2,
                                                    RSFK_typepoints* points,
                                                    int D,
                                                    int N, int lane);

#endif


// Assign a bucket (leaf in the tree) to each warp and a point to each thread (persistent kernel)
__global__
void compute_knn_from_buckets_perwarp_coalesced(int* points_parent,
                              int* points_depth,
                              int* accumulated_nodes_count,
                              RSFK_typepoints* points,
                              int* node_idx_to_leaf_idx,
                              int* nodes_bucket,
                              int* bucket_size,
                              int* knn_indices,
                              RSFK_typepoints* knn_sqr_dist,
                              int N, int D, int max_bucket_size, int K,
                              int MAX_TREE_CHILD, int total_buckets);

// Assign a bucket (leaf in the tree) to each warp and a point to each thread (persistent kernel)
__global__
void compute_knn_from_buckets_perblock_coalesced_symmetric(int* points_parent,
                              int* points_depth,
                              int* accumulated_nodes_count,
                              RSFK_typepoints* points,
                              int* node_idx_to_leaf_idx,
                              int* nodes_bucket,
                              int* bucket_size,
                              int* knn_indices,
                              RSFK_typepoints* knn_sqr_dist,
                              int N, int D, int max_bucket_size, int K,
                              int MAX_TREE_CHILD, int total_buckets);



// Assign a bucket (leaf in the tree) to each warp and a point to each thread (persistent kernel)
__global__
void compute_knn_from_buckets_perblock_coalesced_symmetric_dividek(
                              RSFK_typepoints* points,
                              int* nodes_bucket,
                              int* bucket_size,
                              int* knn_indices,
                              RSFK_typepoints* knn_sqr_dist,
                              int N, int D, int max_bucket_size, int K,
                              int MAX_TREE_CHILD, int total_buckets);

// Assign a bucket (leaf in the tree) to each warp and a point to each thread (persistent kernel)
__global__
void compute_knn_from_buckets_pertile_coalesced_symmetric(int* points_parent,
                              int* points_depth,
                              int* accumulated_nodes_count,
                              RSFK_typepoints* points,
                              int* node_idx_to_leaf_idx,
                              int* nodes_bucket,
                              int* bucket_size,
                              int* knn_indices,
                              RSFK_typepoints* knn_sqr_dist,
                              int N, int D, int max_bucket_size, int K,
                              int MAX_TREE_CHILD, int total_buckets);

#endif