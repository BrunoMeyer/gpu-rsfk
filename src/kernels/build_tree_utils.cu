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

#ifndef __BUILD_TREE_UTILS__CU
#define __BUILD_TREE_UTILS__CU

#include "../include/common.h"

__global__
void
build_tree_utils(int* actual_depth,
                 int* depth_level_count,
                 int* count_new_nodes,
                 int* tree_count,
                 int* accumulated_nodes_count,
                 int* child_count,
                 int* accumulated_child_count,
                 int* device_active_points_count)
// Executed at end of each iteration of tree building
{
    depth_level_count[*actual_depth] = *count_new_nodes;
    accumulated_nodes_count[*actual_depth] = accumulated_nodes_count[*actual_depth-1] + *count_new_nodes;
    
    *actual_depth = *actual_depth+1;
    *tree_count = 0;
    *device_active_points_count = 0;
}


__global__
void
build_tree_fix(int* depth_level_count,
               int* tree_count,
               int* accumulated_nodes_count,
               int max_depth)
{
    *tree_count = 0;
    for(int i=0; i < max_depth; ++i){
        accumulated_nodes_count[i] = *tree_count;
        *tree_count += depth_level_count[i];
    }
}

__global__
void
build_tree_max_leaf_size(int* max_leaf_size,
                         int* min_leaf_size,
                         int* total_leafs,
                         bool* is_leaf,
                         int* child_count,
                         int* count_nodes,
                         int* depth_level_count,
                         int depth)
{
    int tid = blockDim.x*blockIdx.x+threadIdx.x;
    for(int node=tid; node < depth_level_count[depth]; node+=blockDim.x*gridDim.x){
        if(is_leaf[node]){
            atomicMax(max_leaf_size, child_count[node]);
            atomicMin(min_leaf_size, child_count[node]);
            atomicAdd(total_leafs, 1);
        }
    }
}

__global__
void
build_tree_set_leaves_idx(int* leaf_idx_to_node_idx,
                         int* node_idx_to_leaf_idx,
                         int* total_leaves,
                         bool* is_leaf,
                         int* depth_level_count,
                         int* accumulated_nodes_count,
                         int depth)
{
    int tid = blockDim.x*blockIdx.x+threadIdx.x;
    int new_idx;
    int acc_idx;
    for(int node=tid; node < depth_level_count[depth]; node+=blockDim.x*gridDim.x){
        if(is_leaf[node]){
            acc_idx = accumulated_nodes_count[depth] + node;
            new_idx = atomicAdd(total_leaves, 1);
            leaf_idx_to_node_idx[new_idx] = acc_idx;
            node_idx_to_leaf_idx[acc_idx] = new_idx;
        }
    }
}

__global__
void
debug_count_hist_leaf_size(int* hist,
                           int* max_leaf_size,
                           int* min_leaf_size,
                           int* total_leafs,
                           bool* is_leaf,
                           int* child_count,
                           int* count_nodes,
                           int* depth_level_count,
                           int depth)
{
    int tid = blockDim.x*blockIdx.x+threadIdx.x;

    for(int node=tid; node < depth_level_count[depth]; node+=blockDim.x*gridDim.x){
        if(is_leaf[node]){
            atomicAdd(&hist[child_count[node]-*min_leaf_size], 1);
        }
    }
}

#endif