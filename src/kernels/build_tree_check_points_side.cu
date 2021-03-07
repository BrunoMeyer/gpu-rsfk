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

#ifndef __BUILD_TREE_CHECK_POINTS_SIDE__CU
#define __BUILD_TREE_CHECK_POINTS_SIDE__CU

#include "../include/common.h"



__device__
inline
RSFK_typepoints check_hyperplane_side(int node_idx, int p, RSFK_typepoints* tree,
                                           RSFK_typepoints* points, int D, int N,
                                           int* count_new_nodes,
                                           int tidw)
{
    RSFK_typepoints s = 0.0f;
    for(int i=tidw; i < D; i+=32){
        s+=tree[get_tree_idx(node_idx,i,*count_new_nodes,D)]*points[get_point_idx(p,i,N,D)];
    }

    s += __shfl_xor_sync( 0xffffffff, s,  1); // assuming warpSize=32
    s += __shfl_xor_sync( 0xffffffff, s,  2); // assuming warpSize=32
    s += __shfl_xor_sync( 0xffffffff, s,  4); // assuming warpSize=32
    s += __shfl_xor_sync( 0xffffffff, s,  8); // assuming warpSize=32
    s += __shfl_xor_sync( 0xffffffff, s, 16); // assuming warpSize=32
    // lane 0 stores result in SHM
    return s;
}

__global__
void build_tree_check_active_points(int* points_parent,
                                    int* points_depth,
                                    bool* is_leaf,
                                    int* actual_depth,
                                    int* active_points,
                                    int* active_points_count,
                                    int N)
{
    int p;
    int pa_id;
    int tid = blockDim.x*blockIdx.x+threadIdx.x;
    for(p = tid; p < N; p+=blockDim.x*gridDim.x){
        if(points_depth[p] >= *actual_depth-1 && !is_leaf[points_parent[p]]){
            pa_id = atomicAdd(active_points_count, 1);
            active_points[pa_id] = p;
        }
    }
}

__global__
void build_tree_check_points_side(RSFK_typepoints* tree,
                                  int* tree_parents,
                                  int* tree_children,
                                  int* points_parent,
                                  int* points_depth,
                                  int* is_right_child,
                                  bool* is_leaf,
                                  int* child_count,
                                  RSFK_typepoints* points,
                                  int* actual_depth,
                                  int* tree_count,
                                  int* depth_level_count,
                                  int* accumulated_child_count,
                                  int* count_points_on_leafs,
                                  int* sample_candidate_points,
                                  int* points_id_on_sample,
                                  int* active_points,
                                  int* active_points_count,
                                  int* count_new_nodes,
                                  int N, int D, int RANDOM_SEED)
{
    int tid = blockDim.x*blockIdx.x+threadIdx.x;

    int i, j, p, tmp_p;
    int csi;

    RSFK_typepoints product, tmp_product; 

    int tidw = threadIdx.x % 32; // my id on warp

    int k;

    // Set nodes parent in the new depth
    for(i = tid; __any_sync(__activemask(), i < *active_points_count); i+=blockDim.x*gridDim.x){
        p = -1;
        __syncthreads();
        if(i < *active_points_count) p = active_points[i];
        __syncthreads();
        
        for(j=0; j < 32; ++j){
            tmp_p = __shfl_sync(__activemask(), p, j);
            if(tmp_p == -1) continue;
            
            // tmp_product = check_hyperplane_side(points_parent[tmp_p], tmp_p, tree, points, D, N,
            //                                               count_new_nodes,
            //                                               tidw);

            tmp_product = 0.0f;
            for(k=tidw; k < D; k+=32){
                tmp_product+=tree[get_tree_idx(points_parent[tmp_p],k,*count_new_nodes,D)]*points[get_point_idx(tmp_p,k,N,D)];
            }
            
            tmp_product += __shfl_xor_sync( 0xffffffff, tmp_product,  1); // assuming warpSize=32
            tmp_product += __shfl_xor_sync( 0xffffffff, tmp_product,  2); // assuming warpSize=32
            tmp_product += __shfl_xor_sync( 0xffffffff, tmp_product,  4); // assuming warpSize=32
            tmp_product += __shfl_xor_sync( 0xffffffff, tmp_product,  8); // assuming warpSize=32
            tmp_product += __shfl_xor_sync( 0xffffffff, tmp_product, 16); // assuming warpSize=32
                
            if(j == tidw) product = tmp_product;
        }
        __syncwarp();
        if(p == -1) continue;
        
        is_right_child[p] = product < tree[get_tree_idx(points_parent[p],D,*count_new_nodes,D)];
        
        csi = atomicAdd(&count_points_on_leafs[2*points_parent[p]+is_right_child[p]], 1);
        points_id_on_sample[p] = csi;
    }
}

#endif