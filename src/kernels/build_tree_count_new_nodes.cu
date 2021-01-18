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

#ifndef __BUILD_TREE_COUNT_NEW_NODES__CU
#define __BUILD_TREE_COUNT_NEW_NODES__CU

#include "../include/common.h"

__global__
void build_tree_count_new_nodes(RSFK_typepoints* tree,
                                int* tree_parents,
                                int* tree_children,
                                int* points_parent,
                                int* is_right_child,
                                bool* is_leaf,
                                int* count_points_on_leafs,
                                int* child_count,
                                RSFK_typepoints* points,
                                int* actual_depth,
                                int* tree_count,
                                int* depth_level_count,
                                int* count_new_nodes,
                                int N, int D,
                                int MIN_TREE_CHILD, int MAX_TREE_CHILD)
{
    int tid = blockDim.x*blockIdx.x+threadIdx.x;
    for(int node_thread = tid; node_thread < depth_level_count[*actual_depth-1]; node_thread+=blockDim.x*gridDim.x){
        if(!is_leaf[node_thread] && child_count[node_thread] > 0){
            // The representation of nodes are sparse, that is, if at least
            // one node is created, we also create his brother
            if(count_points_on_leafs[2*node_thread] > 0 || count_points_on_leafs[2*node_thread+1] > 0) atomicAdd(count_new_nodes, 2);
            
            // printf("%s: line %d : %d %d %d\n", __FILE__, __LINE__, node_thread, is_leaf[node_thread], depth_level_count[*actual_depth-1]);
        }
    }
}

__global__
void build_tree_accumulate_child_count(int* depth_level_count,
                                       int* count_new_nodes,
                                       int* count_points_on_leafs,
                                       int* accumulated_child_count,
                                       int* actual_depth)
{
    accumulated_child_count[0] = 0;
    for(int i=1; i < 2*depth_level_count[*actual_depth-1]; ++i){
        accumulated_child_count[i] = accumulated_child_count[i-1] + count_points_on_leafs[i-1];
    }
}

__global__
void build_tree_organize_sample_candidates(int* points_parent,
                                           int* points_depth,
                                           bool* is_leaf,
                                           int* is_right_child,
                                           int* count_new_nodes,
                                           int* count_points_on_leafs,
                                           int* accumulated_child_count,
                                           int* sample_candidate_points,
                                           int* points_id_on_sample,
                                           int* actual_depth,
                                           int N)
{
    

    int tid = blockDim.x*blockIdx.x+threadIdx.x;

    int p, is_right;
    int csi;
    
    // Set nodes parent in the new depth
    for(p = tid; p < N; p+=blockDim.x*gridDim.x){
        if(points_depth[p] < *actual_depth-1 || is_leaf[points_parent[p]]) continue;
        csi = points_id_on_sample[p];
        is_right = is_right_child[p];
        sample_candidate_points[accumulated_child_count[2*points_parent[p]+is_right]+csi] = p;
    }
}

#endif