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

#ifndef __RSFK__H
#define __RSFK__H

#include "common.h"

#include "nvgraph.h"

// Thrust includes
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/random.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/functional.h>
#include <thrust/fill.h>
#include <thrust/gather.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>

// CUDA includes
#include <curand.h>
#include <curand_kernel.h>




#include <iostream> 
#include <stdio.h>
#include <cstdlib>
#include <cmath>
#include <bits/stdc++.h> 

#include "common.h"

#include "kernels/build_tree_bucket_points.h"
#include "kernels/build_tree_check_points_side.h"
#include "kernels/build_tree_count_new_nodes.h"
#include "kernels/build_tree_create_nodes.h"
#include "kernels/build_tree_init.h"
#include "kernels/build_tree_update_parents.h"
#include "kernels/build_tree_utils.h"
#include "kernels/compute_knn_from_buckets.h"
#include "kernels/nearest_neighbors_exploring.h"



class TreeInfo{
public:
     int total_leaves;
     int max_child;

     thrust::device_vector<int> device_nodes_buckets;
     thrust::device_vector<int> device_bucket_sizes;
     
     TreeInfo(){}

     TreeInfo(int total_leaves,
              int max_child,
              thrust::device_vector<int> &_device_nodes_buckets,
              thrust::device_vector<int> &_device_bucket_sizes):
              total_leaves(total_leaves),
              max_child(max_child)
     {
          device_nodes_buckets = _device_nodes_buckets;
          device_bucket_sizes = _device_bucket_sizes;
     }

     void free(){
          device_nodes_buckets.clear();
          device_nodes_buckets.shrink_to_fit();
          device_bucket_sizes.clear();
          device_bucket_sizes.shrink_to_fit();
     }
};

// Class used to construct Random Projection forest and execute KNN
class RSFK
{
public:
    
    // Data points. It will be stored as POINTS x DIMENSION or
    // DIMENSION x POINTS considering the defined POINTS_STRUCTURE
    RSFK_typepoints* points; 
    
    // Indices of the estimated k-nearest neighbors for each point
    // and squared distances between
    // It can be previously initialized before the execution of RSFK
    // with valid indices and distances, otherwise it must assume that indices
    // have -1 value and distances FLT_MAX (or DBL_MAX)
    // The indices ARE NOT sorted by the relative distances
    int* knn_indices;
    RSFK_typepoints* knn_sqr_distances;
    
    // Maximum and Minimum number of points that will be present in each leaf node (bucket)
    // This affect the local KNN step after the construction of each tree
    int MIN_TREE_CHILD;
    int MAX_TREE_CHILD;

    // The limit of the tree depth. The algorithm may break if a tree reach
    // a higher value than this parameter. When the tree is 'ready',
    // there is a early stop and this limit will not be reached
    // Known Bug: It may be necessary to add a 'random motion' before the
    // execution of the algorithm to prevent that two points be positioned
    // in the the same location, leading to the impossibility of the
    // tree construction and the reach of MAX_DEPTH 
    int MAX_DEPTH;

    // SEED used to the pseudorandom number generator
    int RANDOM_SEED;

    // Nearest neighbor exploring is a pos-processing technique that will improve
    // the accuracy of the approximated KNN. The neighbors of the neighbors
    // will be treated as neighbor candidates for each point, which is a
    // O(N*D*(K^2)) time cost step. This can be executed several times using the
    // updated indices as the new input, and this ammount is represented by
    // nn_exploring_factor. When nn_exploring_factor=0 the Nearest Neighbor Exploring
    // will not be executed
    int nn_exploring_factor;
    
    RSFK(RSFK_typepoints* points,
         int* knn_indices,
         RSFK_typepoints* knn_sqr_distances,
         int MIN_TREE_CHILD,
         int MAX_TREE_CHILD,
         int MAX_DEPTH,
         int RANDOM_SEED,
         int nn_exploring_factor):
         points(points),
         knn_indices(knn_indices),
         knn_sqr_distances(knn_sqr_distances),
         MIN_TREE_CHILD(MIN_TREE_CHILD),
         MAX_TREE_CHILD(MAX_TREE_CHILD),
         MAX_DEPTH(MAX_DEPTH),
         RANDOM_SEED(RANDOM_SEED),
         nn_exploring_factor(nn_exploring_factor){}
    

    // Create a random projection tree and update the indices by considering
    // the points in the same leaf node as candidates to neighbor
    // This ensure that each point will have K valid neighbors indices, which
    // can be very inaccurate.
    // The device_knn_indices parameter can be previously initialized with
    // valid indices or -1 values. If it has valid indices, also will be necessary
    // to add the precomputed squared distances (device_knn_sqr_distances) 
    TreeInfo create_bucket_from_sample_tree(thrust::device_vector<RSFK_typepoints> &device_points,
                                            int N, int D, int VERBOSE,
                                            std::string run_name);

    void update_knn_indice_with_buckets(thrust::device_vector<RSFK_typepoints> &device_points,
                                        thrust::device_vector<int> &device_knn_indices,
                                        thrust::device_vector<RSFK_typepoints> &device_knn_sqr_distances,
                                        int K, int N, int D, int VERBOSE, TreeInfo tinfo,
                                        std::string run_name);
    
    // Run n_tree times the add_random_projection_tree procedure and the nearest
    // neighbors exploring if necessary
    void knn_gpu_rsfk_forest(int n_trees,
                             int K, int N, int D, int VERBOSE,
                             std::string run_name);

    // Run n_tree times the add_random_projection_tree procedure and the nearest
    // neighbors exploring if necessary
    TreeInfo cluster_by_sample_tree(int N, int D, int VERBOSE,
                                    int** nodes_buckets,
                                    int** bucket_sizes,
                                    std::string run_name);

    int create_cluster_with_hbgf(int* result, int n_trees,
                                 int N, int D, int VERBOSE,
                                 int K, int n_eig_vects,
                                 std::string run_name);

    int spectral_clustering_with_knngraph(int* result, int num_neighbors,
                                          int N, int D, int VERBOSE,
                                          int K, int n_eig_vects,
                                          bool free_knn_indices,
                                          std::string run_name);  
};

#endif