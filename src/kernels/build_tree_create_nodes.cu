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

#ifndef __BUILD_TREE_CREATE_NODES__CU
#define __BUILD_TREE_CREATE_NODES__CU

#include "../include/common.h"

__global__
void build_tree_create_nodes(RSFK_typepoints* tree_new_depth,
                             int* tree_parents_new_depth,
                             int* tree_children,
                             int* points_parent,
                             int* points_depth,
                             int* is_right_child,
                             bool* is_leaf,
                             bool* is_leaf_new_depth,
                             int* child_count,
                             RSFK_typepoints* points,
                             int* actual_depth,
                             int* tree_count,
                             int* depth_level_count,
                             int* count_new_nodes,
                             int* accumulated_nodes_count,
                             int* accumulated_child_count,
                             int* count_points_on_leafs,
                             int* sample_candidate_points,
                             int N, int D, int MIN_TREE_CHILD, int MAX_TREE_CHILD, int RANDOM_SEED)
{
    int tid = blockDim.x*blockIdx.x+threadIdx.x;
    int tidw = threadIdx.x % 32; // my id on warp
    
    curandState_t r; 
    curand_init(*actual_depth*(RANDOM_SEED+blockDim.x)+RANDOM_SEED+tid, // the seed controls the sequence of random values that are produced
            blockIdx.x,  // the sequence number is only important with multiple cores 
            tid,  // the offset is how much extra we advance in the sequence for each call, can be 0 
            &r);

    int p1, p2, node_thread;
    // int parent_id;
    int rand_id;
    
    int i, tmp_p1, tmp_p2, node_idx, tmp_node_idx, is_leaf_node;

    // Create new nodes
    // for(node_thread = tid; node_thread < depth_level_count[*actual_depth-1]; node_thread+=blockDim.x*gridDim.x){
    for(node_thread = tid; __any_sync(__activemask(), node_thread < depth_level_count[*actual_depth-1]); node_thread+=blockDim.x*gridDim.x){
        for(int is_right=0; is_right < 2; ++is_right){
            p1 = -1;
            is_leaf_node = 1;
            
            if(node_thread < depth_level_count[*actual_depth-1]){
                is_leaf_node = is_leaf[node_thread];
            }

            __syncthreads();
            if(!is_leaf_node){
            // if(!is_leaf[node_thread]){
                if(count_points_on_leafs[2*node_thread+is_right] > 0){
                    
                    rand_id = (curand(&r) % count_points_on_leafs[2*node_thread+is_right]);
                    p1 = sample_candidate_points[accumulated_child_count[2*node_thread+is_right]  +  rand_id];
                    rand_id = (curand(&r) % count_points_on_leafs[2*node_thread+is_right]);
                    p2 = sample_candidate_points[accumulated_child_count[2*node_thread+is_right]  +  rand_id];
                    
                    while(p1 == p2 && count_points_on_leafs[2*node_thread+is_right] >= MIN_TREE_CHILD){
                        rand_id = (curand(&r) % count_points_on_leafs[2*node_thread+is_right]);
                        p2 = sample_candidate_points[accumulated_child_count[2*node_thread+is_right]  + rand_id];
                    }

                    node_idx = atomicAdd(tree_count, 1);
                    tree_parents_new_depth[node_idx] = node_thread;
                    tree_children[2*node_thread+is_right] = node_idx;
                }
            }
            __syncthreads();
            // create_node(node_thread, is_right, tree_new_depth, tree_parents_new_depth,
            //             tree_children, tree_count, count_new_nodes, p1, p2, points, D, N);node_thread
            for(i=0; i < 32; ++i){
                tmp_p1 = __shfl_sync(__activemask(), p1, i);
                if(tmp_p1 == -1) continue;
                tmp_p2 = __shfl_sync(__activemask(), p2, i);
                tmp_node_idx = __shfl_sync(__activemask(), node_idx, i);
                // printf("%d %d %d\n", tmp_p1, tmp_2, tmp_node_idx)
                create_node(tree_new_depth, tree_parents_new_depth,
                            tree_children, tree_count, count_new_nodes, tmp_p1, tmp_p2, points, D, N, tmp_node_idx, tidw);
            }
        }
    }
}



/*
// TODO: SPLIT BY PROJECTION INTO A RANDOM DIRECTION

__global__
void init_random_directions(RSFK_typepoints* random_directions,
                            int random_directions_size,
                            int* actual_depth,
                            int RANDOM_SEED)
{
    int tid = blockDim.x*blockIdx.x+threadIdx.x;
    curandState_t r; 
    curand_init(*actual_depth*(RANDOM_SEED+blockDim.x)+RANDOM_SEED+tid, // the seed controls the sequence of random values that are produced
            blockIdx.x,  // the sequence number is only important with multiple cores 
            tid,  // the offset is how much extra we advance in the sequence for each call, can be 0 
            &r);

    for(int i = tid; i < random_directions_size; i+=blockDim.x*gridDim.x){
        random_directions[i] = 2*curand_uniform(&r) - 1.0f;
    }
}

__global__
void project_active_points(RSFK_typepoints* projection_values,
                           RSFK_typepoints* points,
                           int* active_points,
                           int* active_points_count,
                           int* points_parent,
                           int* is_right_child,
                           int* sample_candidate_points,
                           RSFK_typepoints* min_random_proj_values,
                           RSFK_typepoints* max_random_proj_values,
                           int N, int D)
{
    int tid = blockDim.x*blockIdx.x+threadIdx.x;
    int tidw = threadIdx.x % 32; // my id on warp

    int i,j,k;
    int p, tmp_p;
    int is_right;
    int node_idx, init_projection;

    __shared__ RSFK_typepoints product_threads[1024];
    int init_warp_on_block = threadIdx.x-tidw;

    for(i = tid; __any_sync(__activemask(), i < *active_points_count); i+=blockDim.x*gridDim.x){
        __syncthreads();
        
        p = -1;
        if(i < *active_points_count){
            p = active_points[i];
            is_right = is_right_child[p];
            node_idx = (2*points_parent[p]+is_right);
            init_projection = D*node_idx;
        }

        product_threads[init_warp_on_block + tidw] = 0.0f;
        
        __syncthreads();
        for(j=0; j < 32; ++j){
            tmp_p = __shfl_sync(__activemask(), p, j);
            if(tmp_p == -1) continue;
            __syncthreads();
            for(k=tidw; k < D; k+=32){
                product_threads[init_warp_on_block+j]+= projection_values[init_projection+k]*points[get_point_idx(tmp_p,k,N,D)];
            }
        }
        if(p == -1) continue;
        // atomicMin(&min_random_proj_values[node_idx],product_threads[init_warp_on_block+j]);
        // atomicMax(&max_random_proj_values[node_idx],product_threads[init_warp_on_block+j]);
    }
}

__global__
void choose_points_to_split(RSFK_typepoints* projection_values,
                           int* points_parent,
                           int* active_points,
                           int* active_points_count,
                           int* is_right_child,
                           int* sample_candidate_points,
                           RSFK_typepoints* min_random_proj_values,
                           RSFK_typepoints* max_random_proj_values,
                           int N, int D)
{
    int tid = blockDim.x*blockIdx.x+threadIdx.x;
    int tidw = threadIdx.x % 32; // my id on warp

    int i,j,k;
    int p, tmp_p;
    int is_right;
    int node_idx;

    __shared__ RSFK_typepoints product_threads[1024];
    int init_warp_on_block = threadIdx.x-tidw;

    for(i = tid; i < *active_points_count; i+=blockDim.x*gridDim.x){
        __syncthreads();
        
        p = active_points[i];
        is_right = is_right_child[p];
        node_idx = (2*points_parent[p]+is_right);

        // min_random_proj_values[node_idx]
        // atomicMax(&max_random_proj_values[node_idx],product_threads[init_warp_on_block+j]);
    }
}

__global__
void build_tree_create_nodes_random_projection(RSFK_typepoints* tree_new_depth,
                             int* tree_parents_new_depth,
                             int* tree_children,
                             int* points_parent,
                             int* points_depth,
                             int* is_right_child,
                             bool* is_leaf,
                             bool* is_leaf_new_depth,
                             int* child_count,
                             RSFK_typepoints* points,
                             int* actual_depth,
                             int* tree_count,
                             int* depth_level_count,
                             int* count_new_nodes,
                             int* accumulated_nodes_count,
                             int* accumulated_child_count,
                             int* count_points_on_leafs,
                             int* sample_candidate_points,
                             int* random_directions,
                             int N, int D, int K, int MAX_TREE_CHILD, int RANDOM_SEED)
{
    int tid = blockDim.x*blockIdx.x+threadIdx.x;
    curandState_t r; 
    curand_init(*actual_depth*(RANDOM_SEED+blockDim.x)+RANDOM_SEED+tid, // the seed controls the sequence of random values that are produced
            blockIdx.x,  // the sequence number is only important with multiple cores 
            tid,  // the offset is how much extra we advance in the sequence for each call, can be 0 
            &r);

    int p1, p2, node_thread;
    // int parent_id;
    int rand_id;

    float random_dir;
    int min_proj_point;
    
    // Create new nodes
    for(node_thread = tid; node_thread < depth_level_count[*actual_depth-1]; node_thread+=blockDim.x*gridDim.x){
        for(int is_right=0; is_right < 2; ++is_right){
            if(!is_leaf[node_thread]){
                if(count_points_on_leafs[2*node_thread+is_right] > 0){
                    random_dir = 2*curand_uniform(&r)-1.0f;
                    rand_id = (curand(&r) % count_points_on_leafs[2*node_thread+is_right]);
                    p1 = sample_candidate_points[accumulated_child_count[2*node_thread+is_right]  +  rand_id];
                    rand_id = (curand(&r) % count_points_on_leafs[2*node_thread+is_right]);
                    p2 = sample_candidate_points[accumulated_child_count[2*node_thread+is_right]  +  rand_id];
                    
                    while(p1 == p2 && count_points_on_leafs[2*node_thread+is_right] > K){
                        rand_id = (curand(&r) % count_points_on_leafs[2*node_thread+is_right]);
                        p2 = sample_candidate_points[accumulated_child_count[2*node_thread+is_right]  + rand_id];
                    }

                    __syncthreads();
                    create_node(node_thread, is_right, tree_new_depth, tree_parents_new_depth,
                                tree_children, tree_count, count_new_nodes, p1, p2, points, D, N);
                }
            }
        }
    }
}
*/
#endif