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


// Implementation file for the python extensions

// #include "pymodule_ext.h"

#include <sys/types.h>
#include "../rsfk.cu"
#include <thrust/fill.h>

extern "C" {
    void pymodule_rsfk_knn(int n_trees,
                           int num_neighbors,
                           int N,
                           int D,
                           int MIN_TREE_CHILD,
                           int MAX_TREE_CHILD,
                           int MAX_DEPTH,
                           int VERBOSE,
                           int RANDOM_STATE,
                           int nn_exploring_factor,
                           RSFK_typepoints* points,
                           int* knn_indices,
                           RSFK_typepoints* knn_sqr_distances);

    void pymodule_cluster_by_sample_tree(int N,
                                         int D,
                                         int MIN_TREE_CHILD,
                                         int MAX_TREE_CHILD,
                                         int MAX_DEPTH,
                                         int VERBOSE,
                                         int RANDOM_STATE,
                                         RSFK_typepoints* points,
                                         int** nodes_buckets,
                                         int** bucket_sizes,
                                         int* total_leaves,
                                         int* max_child);

    void pymodule_create_cluster_with_hbgf(int n_trees,
                                           int K,
                                           int n_eig_vects,
                                           int N,
                                           int D,
                                           int MIN_TREE_CHILD,
                                           int MAX_TREE_CHILD,
                                           int MAX_DEPTH,
                                           int VERBOSE,
                                           int RANDOM_STATE,
                                           RSFK_typepoints* points,
                                           int* result);

    void pymodule_spectral_clustering_with_knngraph(int n_trees,
                                                    int K, //number of clusters
                                                    int num_neighbors,
                                                    int nn_exploring_factor,
                                                    int n_eig_vects,
                                                    int N,
                                                    int D,
                                                    int MIN_TREE_CHILD,
                                                    int MAX_TREE_CHILD,
                                                    int MAX_DEPTH,
                                                    int VERBOSE,
                                                    int RANDOM_STATE,
                                                    RSFK_typepoints* points,
                                                    int* result);
}

void pymodule_rsfk_knn(int n_trees,
                       int num_neighbors,
                       int N,
                       int D,
                       int MIN_TREE_CHILD,
                       int MAX_TREE_CHILD,
                       int MAX_DEPTH,
                       int VERBOSE,
                       int RANDOM_STATE,
                       int nn_exploring_factor,
                       RSFK_typepoints* points,
                       int* knn_indices,
                       RSFK_typepoints* knn_sqr_distances)
{
    std::string run_name="run";
    RSFK rsfk_knn(points, knn_indices, knn_sqr_distances,
                  MIN_TREE_CHILD, MAX_TREE_CHILD,
                  MAX_DEPTH, RANDOM_STATE, nn_exploring_factor);

    rsfk_knn.knn_gpu_rsfk_forest(n_trees, num_neighbors, N, D, VERBOSE, run_name);
}

void pymodule_cluster_by_sample_tree(int N,
                                     int D,
                                     int MIN_TREE_CHILD,
                                     int MAX_TREE_CHILD,
                                     int MAX_DEPTH,
                                     int VERBOSE,
                                     int RANDOM_STATE,
                                     RSFK_typepoints* points,
                                     int** nodes_buckets,
                                     int** bucket_sizes,
                                     int* total_leaves,
                                     int* max_child)
{
    std::string run_name="run";

    RSFK rsfk_knn(points, nullptr, nullptr,
                  MIN_TREE_CHILD, MAX_TREE_CHILD,
                  MAX_DEPTH, RANDOM_STATE, 0);

    TreeInfo tinfo;

    tinfo = rsfk_knn.cluster_by_sample_tree(N, D, VERBOSE,
                                            nodes_buckets,
                                            bucket_sizes,
                                            run_name);

    *total_leaves = tinfo.total_leaves;
    *max_child = tinfo.max_child;
}


void pymodule_create_cluster_with_hbgf(int n_trees,
                                       int K,
                                       int n_eig_vects,
                                       int N,
                                       int D,
                                       int MIN_TREE_CHILD,
                                       int MAX_TREE_CHILD,
                                       int MAX_DEPTH,
                                       int VERBOSE,
                                       int RANDOM_STATE,
                                       RSFK_typepoints* points,
                                       int* result)
{
    std::string run_name="run";

    RSFK rsfk_knn(points, nullptr, nullptr,
                  MIN_TREE_CHILD, MAX_TREE_CHILD,
                  MAX_DEPTH, RANDOM_STATE, 0);

    int err;
    err = rsfk_knn.create_cluster_with_hbgf(result,
                                            n_trees,
                                            N, D, VERBOSE,
                                            K, n_eig_vects,
                                            run_name);

}


void pymodule_spectral_clustering_with_knngraph(int n_trees,
                                                int K, //number of clusters
                                                int num_neighbors,
                                                int nn_exploring_factor,
                                                int n_eig_vects,
                                                int N,
                                                int D,
                                                int MIN_TREE_CHILD,
                                                int MAX_TREE_CHILD,
                                                int MAX_DEPTH,
                                                int VERBOSE,
                                                int RANDOM_STATE,
                                                RSFK_typepoints* points,
                                                int* result)
{
    std::string run_name="run";

    int* knn_indices     = new int[N*num_neighbors];
    float* knn_distances = new float[N*num_neighbors];
    thrust::fill(knn_indices, knn_indices+ N*num_neighbors, -1);
    thrust::fill(knn_distances, knn_distances+ N*num_neighbors, FLT_MAX);

    RSFK rsfk_knn(points, knn_indices, knn_distances,
                  MIN_TREE_CHILD, MAX_TREE_CHILD,
                  MAX_DEPTH, RANDOM_STATE, nn_exploring_factor);

    rsfk_knn.knn_gpu_rsfk_forest(n_trees, num_neighbors,
                                 N, D, VERBOSE, run_name);

    int err;
    err = rsfk_knn.spectral_clustering_with_knngraph(result,
                                                     num_neighbors,
                                                     N, D, VERBOSE,
                                                     K, n_eig_vects,
                                                     true,
                                                     run_name);
    // delete [] knn_indices;
    delete [] knn_distances;
}