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

#ifndef __BUILD_TREE_INIT__CU
#define __BUILD_TREE_INIT__CU

#include "../include/common.h"

__global__
void build_tree_init(RSFK_typepoints* tree,
                     int* tree_parents,
                     int* tree_children,
                     int* points_parent,
                     int* child_count,
                     bool* is_leaf,
                     RSFK_typepoints* points,
                     int* actual_depth,
                     int* tree_count,
                     int* depth_level_count,
                     int* accumulated_nodes_count,
                     int* accumulated_child_count,
                     int* device_count_points_on_leafs,
                     int* count_new_nodes,
                     int N, int D, int RANDOM_SEED)
{
    int tid = blockDim.x*blockIdx.x+threadIdx.x;
    curandState_t r; 
    curand_init(RANDOM_SEED+tid, // the seed controls the sequence of random values that are produced
            blockIdx.x,  // the sequence number is only important with multiple cores 
            tid,  // the offset is how much extra we advance in the sequence for each call, can be 0 
            &r);

    int p1, p2;
    // Sample two random points
    if(tid == 0){
        p1 = curand(&r) % N;
        p2 = p1;
        // Ensure that two different points was sampled
        while(p1 == p2 && N > 1){
            p2 = curand(&r) % N;
        }
        *count_new_nodes = 1;
        
        create_root(tree, tree_parents, tree_children, tree_count, p1, p2,
                    count_new_nodes, points, D, N);
        depth_level_count[0] = 1;
        accumulated_nodes_count[0] = 0;
        is_leaf[0] = false;
        child_count[0] = N;
        accumulated_child_count[0] = 0;
        *actual_depth = 1;
    }
}

#endif