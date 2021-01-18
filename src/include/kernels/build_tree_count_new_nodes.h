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

#ifndef __BUILD_TREE_COUNT_NEW_NODES__H
#define __BUILD_TREE_COUNT_NEW_NODES__H
#include "../common.h"
#include "../../kernels/build_tree_count_new_nodes.cu"

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
                                int MIN_TREE_CHILD, int MAX_TREE_CHILD);


__global__
void build_tree_accumulate_child_count(int* depth_level_count,
                                       int* count_new_nodes,
                                       int* count_points_on_leafs,
                                       int* accumulated_child_count,
                                       int* actual_depth);

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
                                           int N);

#endif