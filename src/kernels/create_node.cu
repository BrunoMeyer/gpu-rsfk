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

#ifndef __CREATE_NODE__CU
#define __CREATE_NODE__CU

#include "../include/common.h"

__device__
inline
void create_root(RSFK_typepoints* tree,
                 int* tree_parents,
                 int* tree_children,
                 int* tree_count,
                 int p1,
                 int p2,
                 int* count_new_nodes,
                 RSFK_typepoints* points,
                 int D, int N)
{
    // Average point
    // node_path*D*2 : D*2 = size of centroid point and normal vector

    int node_idx = 0;
    
    tree_parents[node_idx] = -1;
    *tree_count = 0;

    int i;
    // tree[node_idx*(D+1) + D] = 0.0f;
    tree[get_tree_idx(node_idx,D,*count_new_nodes,D)] = 0.0f;

    for(i=0;i < D; ++i){
        tree[get_tree_idx(node_idx,i,*count_new_nodes,D)] = points[get_point_idx(p1,i,N,D)]-points[get_point_idx(p2,i,N,D)];
        tree[get_tree_idx(node_idx,D,*count_new_nodes,D)]+= tree[get_tree_idx(node_idx,i,*count_new_nodes,D)]*(points[get_point_idx(p1,i,N,D)]+points[get_point_idx(p2,i,N,D)])/2; // multiply the point of plane and the normal vector 
    }
}

__device__
inline
void create_node(int parent,
                 int is_right_child,
                 RSFK_typepoints* tree,
                 int* tree_parents,
                 int* tree_children,
                 int* tree_count,
                 int* count_new_nodes,
                 int p1,
                 int p2,
                 RSFK_typepoints* points,
                 int D, int N)
{
    // Average point
    // node_path*D*2 : D*2 = size of centroid point and normal vector

    int node_idx = atomicAdd(tree_count, 1);
    tree_parents[node_idx] = parent;
    
    tree_children[2*parent+is_right_child] = node_idx;
    int i;

    RSFK_typepoints s = 0.0f;
    for(i=0; i < D; ++i){
        tree[get_tree_idx(node_idx,i,*count_new_nodes,D)] = points[get_point_idx(p1,i,N,D)]-points[get_point_idx(p2,i,N,D)];
        s+= tree[get_tree_idx(node_idx,i,*count_new_nodes,D)]*(points[get_point_idx(p1,i,N,D)]+points[get_point_idx(p2,i,N,D)])/2; // multiply the point of plane and the normal vector 
    }
    tree[get_tree_idx(node_idx,D,*count_new_nodes,D)] = s;
}

#endif