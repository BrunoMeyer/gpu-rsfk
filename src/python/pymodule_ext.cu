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
#include "../rpfk.cu"

extern "C" {
    void pymodule_rpfk_knn(int n_trees,
                           int K,
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
}

void pymodule_rpfk_knn(int n_trees,
                       int K,
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
    RPFK rpfk_knn(points, knn_indices, knn_sqr_distances,
                  MIN_TREE_CHILD, MAX_TREE_CHILD,
                  MAX_DEPTH, RANDOM_STATE, nn_exploring_factor);

    rpfk_knn.knn_gpu_rpfk_forest(n_trees, K, N, D, VERBOSE, run_name);
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

    RPFK rpfk_knn(points, nullptr, nullptr,
                  MIN_TREE_CHILD, MAX_TREE_CHILD,
                  MAX_DEPTH, RANDOM_STATE, 0);

    TreeInfo tinfo;

    tinfo = rpfk_knn.cluster_by_sample_tree(N, D, VERBOSE,
                                            nodes_buckets,
                                            bucket_sizes,
                                            run_name);

    *total_leaves = tinfo.total_leaves;
    *max_child = tinfo.max_child;
}