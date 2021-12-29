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

#ifndef __BUILD_TREE_UPDATE_PARENTS_ANN__CU
#define __BUILD_TREE_UPDATE_PARENTS_ANN__CU

#include "../include/common.h"

__global__
void
build_tree_utils_ann(int* actual_depth,
                 int* depth_level_count,
                 int* count_new_nodes,
                 int* tree_count,
                 int* accumulated_nodes_count,
                 int* device_active_points_count)
// Executed at end of each iteration of tree building
{
    *actual_depth = *actual_depth+1;
    *device_active_points_count = 0;
}

__global__
void build_tree_update_parents_ann(
    RSFK_typepoints* tree,
    int* tree_parents,
    int* tree_children,
    int* points_parent,
    int* points_depth,
    int* is_right_child,
    bool* is_leaf,
    bool* is_leaf_new_depth,
    int* child_count,
    int* child_count_new_depth,
    RSFK_typepoints* points,
    int* actual_depth,
    int* tree_count,
    int* depth_level_count,
    int* count_new_nodes,
    int N, int D,
    int MIN_TREE_CHILD, int MAX_TREE_CHILD)
{
    int tid = blockDim.x*blockIdx.x+threadIdx.x;

    int right_child, p;
    int updated_count;
    // Set nodes parent in the new depth
    for(p = tid; p < N; p+=blockDim.x*gridDim.x){
        if(points_depth[p] == *actual_depth-1 && !is_leaf[points_parent[p]]){
            right_child = is_right_child[p];
            points_parent[p] = tree_children[2*points_parent[p]+right_child];
            points_depth[p] = *actual_depth;
            updated_count = atomicAdd(&child_count_new_depth[points_parent[p]],1)+1;
        }
    }
}

#endif