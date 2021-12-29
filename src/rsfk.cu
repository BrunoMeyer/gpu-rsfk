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

#ifndef __RSFK__CU
#define __RSFK__CU

#include "include/rsfk.h"

static void CudaTest(char* msg)
{
  cudaError_t e;

  cudaDeviceSynchronize();
  if (cudaSuccess != (e = cudaGetLastError())) {
    fprintf(stderr, "%s: %d\n", msg, e);
    fprintf(stderr, "%s\n", cudaGetErrorString(e));
    exit(1);
  }
}

class Cron
{
    public:
        std::chrono::time_point<std::chrono::high_resolution_clock> t_start;
        double t_total;

        Cron(){
            this->reset();
        }
        void reset(){
            t_total = 0.0;
            start();
        }
        void start(){
            t_start = std::chrono::high_resolution_clock::now();
        }
        double stop(){
            std::chrono::time_point<std::chrono::high_resolution_clock> t_end = std::chrono::high_resolution_clock::now();
            double dt = std::chrono::duration<double, std::milli>(t_end-t_start).count();
            t_total+=dt;
            return dt;
        }
};


// void RSFK::ann_build_index_and_query(){
    
// }

void RSFK::knn_gpu_rsfk_forest_ann(int n_trees,
                                   int K, int N, int NQ, int D, int VERBOSE,
                                   std::string run_name,
                                   RSFKIndexTree* rsfkindextree)
{
    Cron init_tree_cron, end_tree_cron, total_tree_build_cron, check_active_points_cron,
         update_parents_cron, create_nodes_cron, check_points_side_cron, tree_count_cron,
         organize_sample_candidate_cron, dynamic_memory_allocation_cron;
    
    thrust::device_vector<RSFK_typepoints> device_points(points, points+N*D);
    thrust::device_vector<RSFK_typepoints> device_query_points(query_points, query_points+NQ*D);


    TreeInfo tinfo;
    ForestLog forest_log = ForestLog(n_trees);

    tinfo = create_bucket_from_sample_tree(
        device_points, N, D, VERBOSE, forest_log, run_name, false, rsfkindextree);
    

    init_tree_cron.start();
    
    // DEBUG Variables
    #if RSFK_COMPILE_TYPE == RSFK_DEBUG
        thrust::device_vector<int> device_count_undo_leaf(1, 0);
    #endif

    // Structures that cost O(N) space

    // The id of the node parent for each point. Note that nodes from
    // different levels of the tree can have the same id 
    thrust::device_vector<int> device_points_parent(NQ, 0);

    // The level of the node parent of each point
    thrust::device_vector<int> device_points_depth(NQ, 0);

    // Flag that indicates if the parent of each point is left (0) or right (1)
    thrust::device_vector<int> device_is_right_child(NQ, 0);

    // Vector that sequentially put points candidates to be selected in the
    // creation of a new hyperplane
    thrust::device_vector<int> device_sample_candidate_points(2*NQ, -1);

    // Random Projection Forest
    // thrust::device_vector<RSFK_typepoints> device_min_random_proj_values(N, FLT_MAX);
    // thrust::device_vector<RSFK_typepoints> device_max_random_proj_values(N, FLT_MIN);

    // The id of each point inside of each subvector of device_sample_candidate_points
    thrust::device_vector<int> device_points_id_on_sample(NQ, -1);

    // Flag that indicates if a point is already inside a leaf (bucket) node
    thrust::device_vector<int> device_active_points(NQ, -1);

    
    // Structures that cost O(MAX_DEPTH) or O(1) space
    // MAX_DEPTH is assumed to be constant

    // An vector with the values of device_depth_level_count as a cumulative sum
    thrust::device_vector<int> device_accumulated_nodes_count(MAX_DEPTH,-1);

    // Count the number of new nodes in the current level
    thrust::device_vector<int> device_count_new_nodes(1,1);

    // Index of the current level of tree
    thrust::device_vector<int> device_actual_depth(1,0);

    // Total of points that are not already in a leaf (bucket)
    thrust::device_vector<int> device_active_points_count(1, 0);


    // Automatically get the ideal number of threads per block and total of blocks
    // used in kernels
    // TODO: Enable user to specify the GPU id to be used
    int devUsed = 0;
    cudaSetDevice(devUsed);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, devUsed);
    const int NT = deviceProp.maxThreadsPerBlock;
    const int NB = deviceProp.multiProcessorCount*(deviceProp.maxThreadsPerMultiProcessor/deviceProp.maxThreadsPerBlock);

    if(VERBOSE >=1){
        std::cout << "Running kernels with " << NT << " threads per block and " << NB << " blocks" << std::endl; 
    }

    if(VERBOSE >= 1){
        std::cout << std::endl;
    }

    
    init_tree_cron.stop();
    total_tree_build_cron.start();
    int depth, count_new_nodes, count_total_nodes, reached_max_depth;
    
    count_total_nodes = 1;
    count_new_nodes = 1;


    for(depth=1; depth < rsfkindextree->reached_max_depth; ++depth){
        check_active_points_cron.start();
        cudaDeviceSynchronize();
        
        build_tree_check_active_points<<<NB,NT>>>(thrust::raw_pointer_cast(device_points_parent.data()),
                                                  thrust::raw_pointer_cast(device_points_depth.data()),
                                                  thrust::raw_pointer_cast(rsfkindextree->device_is_leaf[depth-1]->data()),
                                                  thrust::raw_pointer_cast(device_actual_depth.data()),
                                                  thrust::raw_pointer_cast(device_active_points.data()),
                                                  thrust::raw_pointer_cast(device_active_points_count.data()),
                                                  NQ);
        cudaDeviceSynchronize();
        CudaTest((char *)"build_tree_check_active_points Kernel failed!");
        check_active_points_cron.stop();

        check_points_side_cron.start();
        build_tree_check_points_side<<<NB,NT>>>(
            thrust::raw_pointer_cast(rsfkindextree->device_tree[depth-1]->data()),
            thrust::raw_pointer_cast(rsfkindextree->device_tree_parents[depth-1]->data()),
            thrust::raw_pointer_cast(rsfkindextree->device_tree_children[depth-1]->data()),
            thrust::raw_pointer_cast(device_points_parent.data()),
            thrust::raw_pointer_cast(device_points_depth.data()),
            thrust::raw_pointer_cast(device_is_right_child.data()),
            thrust::raw_pointer_cast(rsfkindextree->device_is_leaf[depth-1]->data()),
            thrust::raw_pointer_cast(rsfkindextree->device_child_count[depth-1]->data()),
            thrust::raw_pointer_cast(device_query_points.data()),
            thrust::raw_pointer_cast(device_actual_depth.data()),
            thrust::raw_pointer_cast(rsfkindextree->device_tree_count->data()),
            thrust::raw_pointer_cast(rsfkindextree->device_depth_level_count->data()),
            thrust::raw_pointer_cast(rsfkindextree->device_accumulated_child_count[depth-1]->data()),
            thrust::raw_pointer_cast(rsfkindextree->device_count_points_on_leaves[depth-1]->data()),
            thrust::raw_pointer_cast(device_sample_candidate_points.data()),
            thrust::raw_pointer_cast(device_points_id_on_sample.data()),
            thrust::raw_pointer_cast(device_active_points.data()),
            thrust::raw_pointer_cast(device_active_points_count.data()),
            thrust::raw_pointer_cast(device_count_new_nodes.data()),
            NQ, D, RANDOM_SEED);
        cudaDeviceSynchronize();
        CudaTest((char *)"build_tree_check_points_side Kernel failed!");
        check_points_side_cron.stop();
        
        
        


        update_parents_cron.start();
        build_tree_update_parents_ann<<<NB,NT>>>(
            thrust::raw_pointer_cast(rsfkindextree->device_tree[depth]->data()),
            thrust::raw_pointer_cast(rsfkindextree->device_tree_parents[depth-1]->data()),
            thrust::raw_pointer_cast(rsfkindextree->device_tree_children[depth-1]->data()),
            thrust::raw_pointer_cast(device_points_parent.data()),
            thrust::raw_pointer_cast(device_points_depth.data()),
            thrust::raw_pointer_cast(device_is_right_child.data()),
            thrust::raw_pointer_cast(rsfkindextree->device_is_leaf[depth-1]->data()),
            thrust::raw_pointer_cast(rsfkindextree->device_is_leaf[depth]->data()),
            thrust::raw_pointer_cast(rsfkindextree->device_child_count[depth-1]->data()),
            thrust::raw_pointer_cast(rsfkindextree->device_child_count[depth]->data()), // new depth is created
            thrust::raw_pointer_cast(device_query_points.data()),
            thrust::raw_pointer_cast(device_actual_depth.data()),
            thrust::raw_pointer_cast(rsfkindextree->device_tree_count->data()),
            thrust::raw_pointer_cast(rsfkindextree->device_depth_level_count->data()),
            thrust::raw_pointer_cast(device_count_new_nodes.data()),
            NQ, D, MIN_TREE_CHILD, MAX_TREE_CHILD);
        cudaDeviceSynchronize();
        CudaTest((char *)"build_tree_update_parents Kernel failed!");
        
        
        
        build_tree_utils_ann<<<1,1>>>(
            thrust::raw_pointer_cast(device_actual_depth.data()),
            thrust::raw_pointer_cast(rsfkindextree->device_depth_level_count->data()),
            thrust::raw_pointer_cast(device_count_new_nodes.data()),
            thrust::raw_pointer_cast(rsfkindextree->device_tree_count->data()),
            thrust::raw_pointer_cast(device_accumulated_nodes_count.data()),
            thrust::raw_pointer_cast(device_active_points_count.data())
        );

        cudaDeviceSynchronize();
        CudaTest((char *)"build_tree_utils Kernel failed!");
        update_parents_cron.stop();
        

        if(VERBOSE >= 2){
            std::cout << "\e[ABuilding Tree Depth: " << depth+1 << "/" << MAX_DEPTH;
            std::cout << " | Total nodes: " << count_total_nodes << std::endl;
        }

        if(count_new_nodes == 0){
            if(VERBOSE >= 1) std::cout << "Early stop: 0 new nodes created." << std::endl;
            break;
        }
    }
    
    
    // do 
    // {
    // std::cout << '\n' << "Press a key to continue...";
    // } while (std::cin.get() != '\n');


    end_tree_cron.stop();

}

TreeInfo RSFK::create_bucket_from_sample_tree(
    thrust::device_vector<RSFK_typepoints> &device_points,
    int N, int D, int VERBOSE,
    ForestLog& forest_log,
    std::string run_name="out.png",
    bool free_index=true,
    RSFKIndexTree* rsfkindextree=nullptr)
{
    Cron init_tree_cron, end_tree_cron, total_tree_build_cron, check_active_points_cron,
         update_parents_cron, create_nodes_cron, check_points_side_cron, tree_count_cron,
         organize_sample_candidate_cron, dynamic_memory_allocation_cron;
    

    init_tree_cron.start();
    
    // They are pointers to each vector that represent a level of the tree
    // These vectors are allocated dynamically during the tree construction
    
    // Tree nodes hyperplanes values (equations of size D+1)
    thrust::device_vector<RSFK_typepoints>** device_tree = (thrust::device_vector<RSFK_typepoints>**) malloc(sizeof(thrust::device_vector<RSFK_typepoints>*)*MAX_DEPTH);
    
    // Random Projection Forest
    // thrust::device_vector<RSFK_typepoints>** device_random_directions = (thrust::device_vector<RSFK_typepoints>**) malloc(sizeof(thrust::device_vector<RSFK_typepoints>*)*MAX_DEPTH);
    
    // Id of the parent (from the last level) of each node at current level
    thrust::device_vector<int>** device_tree_parents = (thrust::device_vector<int>**) malloc(sizeof(thrust::device_vector<int>*)*MAX_DEPTH);

    // Id of the two children (of the next level) of each node at current level
    thrust::device_vector<int>** device_tree_children = (thrust::device_vector<int>**) malloc(sizeof(thrust::device_vector<int>*)*MAX_DEPTH);

    // Flag that indicates if this node is a leaf (bucket)
    thrust::device_vector<bool>** device_is_leaf = (thrust::device_vector<bool>**) malloc(sizeof(thrust::device_vector<bool>*)*MAX_DEPTH);
    
    // Total of points "inside" of each node
    thrust::device_vector<int>** device_child_count = (thrust::device_vector<int>**) malloc(sizeof(thrust::device_vector<int>*)*MAX_DEPTH);
    
    // Vector with the cumulative sum of the number of points "inside" each node
    // The id of each node is used to assume an arbitrary order
    thrust::device_vector<int>** device_accumulated_child_count = (thrust::device_vector<int>**) malloc(sizeof(thrust::device_vector<int>*)*MAX_DEPTH);
    
    // Same that device_child_count. This is used in check_points_side and 
    // device_child_count is used in update_parents
    // TODO: Remove redundant processing 
    thrust::device_vector<int>** device_count_points_on_leaves = (thrust::device_vector<int>**) malloc(sizeof(thrust::device_vector<int>*)*MAX_DEPTH);


    // Allocates the vectors for the first level of the tree
    int MAX_NODES = 1;
    // device_tree[0] = new thrust::device_vector<RSFK_typepoints>(((D+19)+1)*MAX_NODES);
    device_tree[0] = new thrust::device_vector<RSFK_typepoints>((D+1)*MAX_NODES);
    // Random Projection Forest
    // device_random_directions[0] = new thrust::device_vector<RSFK_typepoints>(2*D*MAX_NODES);
    device_tree_parents[0] = new thrust::device_vector<int>(MAX_NODES,-1);
    device_tree_children[0] = new thrust::device_vector<int>(2*MAX_NODES,-1);
    device_is_leaf[0] = new thrust::device_vector<bool>(MAX_NODES, false);
    device_child_count[0] = new thrust::device_vector<int>(MAX_NODES, 0);
    device_accumulated_child_count[0] = new thrust::device_vector<int>(2*MAX_NODES, 0);
    device_count_points_on_leaves[0] = new thrust::device_vector<int>(2*MAX_NODES, 0);



    // Structures that cost O(N) space

    // The id of the node parent for each point. Note that nodes from
    // different levels of the tree can have the same id 
    thrust::device_vector<int> device_points_parent(N, 0);

    // The level of the node parent of each point
    thrust::device_vector<int> device_points_depth(N, 0);

    // Flag that indicates if the parent of each point is left (0) or right (1)
    thrust::device_vector<int> device_is_right_child(N, 0);

    // Vector that sequentially put points candidates to be selected in the
    // creation of a new hyperplane
    thrust::device_vector<int> device_sample_candidate_points(2*N, -1);

    // Random Projection Forest
    // thrust::device_vector<RSFK_typepoints> device_min_random_proj_values(N, FLT_MAX);
    // thrust::device_vector<RSFK_typepoints> device_max_random_proj_values(N, FLT_MIN);

    // The id of each point inside of each subvector of device_sample_candidate_points
    thrust::device_vector<int> device_points_id_on_sample(N, -1);

    // Flag that indicates if a point is already inside a leaf (bucket) node
    thrust::device_vector<int> device_active_points(N, -1);




    // Structures that cost O(MAX_DEPTH) or O(1) space
    // MAX_DEPTH is assumed to be constant

    // Total of nodes inside each level
    // This is used to allocate several structures dynamically
    thrust::device_vector<int> device_depth_level_count(MAX_DEPTH,-1);

    // An vector with the values of device_depth_level_count as a cumulative sum
    thrust::device_vector<int> device_accumulated_nodes_count(MAX_DEPTH,-1);

    // Count the total of nodes allocated in the tree
    thrust::device_vector<int> device_tree_count(1,0);

    // Count the number of new nodes in the current level
    thrust::device_vector<int> device_count_new_nodes(1,1);

    // Index of the current level of tree
    thrust::device_vector<int> device_actual_depth(1,0);

    // Total of points that are not already in a leaf (bucket)
    thrust::device_vector<int> device_active_points_count(1, 0);


    // DEBUG Variables
    #if RSFK_COMPILE_TYPE == RSFK_DEBUG
        thrust::device_vector<int> device_count_undo_leaf(1, 0);
    #endif

    

    // Automatically get the ideal number of threads per block and total of blocks
    // used in kernels
    // TODO: Enable user to specify the GPU id to be used
    int devUsed = 0;
    cudaSetDevice(devUsed);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, devUsed);
    const int NT = deviceProp.maxThreadsPerBlock;
    const int NB = deviceProp.multiProcessorCount*(deviceProp.maxThreadsPerMultiProcessor/deviceProp.maxThreadsPerBlock);

    if(VERBOSE >=1){
        std::cout << "Running kernels with " << NT << " threads per block and " << NB << " blocks" << std::endl; 
    }

    build_tree_init<<<NB,NT>>>(thrust::raw_pointer_cast(device_tree[0]->data()),
                               thrust::raw_pointer_cast(device_tree_parents[0]->data()),
                               thrust::raw_pointer_cast(device_tree_children[0]->data()),
                               thrust::raw_pointer_cast(device_points_parent.data()),
                               thrust::raw_pointer_cast(device_child_count[0]->data()),
                               thrust::raw_pointer_cast(device_is_leaf[0]->data()),
                               thrust::raw_pointer_cast(device_points.data()),
                               thrust::raw_pointer_cast(device_actual_depth.data()),
                               thrust::raw_pointer_cast(device_tree_count.data()),
                               thrust::raw_pointer_cast(device_depth_level_count.data()),
                               thrust::raw_pointer_cast(device_accumulated_nodes_count.data()),
                               thrust::raw_pointer_cast(device_accumulated_child_count[0]->data()),
                               thrust::raw_pointer_cast(device_count_points_on_leaves[0]->data()),
                               thrust::raw_pointer_cast(device_count_new_nodes.data()),
                               N, D, RANDOM_SEED);
    cudaDeviceSynchronize();
    CudaTest((char *)"build_tree_init Kernel failed!");

    if(VERBOSE >= 1){
        std::cout << std::endl;
    }

    
    init_tree_cron.stop();
    total_tree_build_cron.start();
    int depth, count_new_nodes, count_total_nodes, reached_max_depth;
    
    count_total_nodes = 1;
    count_new_nodes = 1;
    for(depth=1; depth < MAX_DEPTH; ++depth){
        check_active_points_cron.start();
        cudaDeviceSynchronize();
        
        build_tree_check_active_points<<<NB,NT>>>(thrust::raw_pointer_cast(device_points_parent.data()),
                                                  thrust::raw_pointer_cast(device_points_depth.data()),
                                                  thrust::raw_pointer_cast(device_is_leaf[depth-1]->data()),
                                                  thrust::raw_pointer_cast(device_actual_depth.data()),
                                                  thrust::raw_pointer_cast(device_active_points.data()),
                                                  thrust::raw_pointer_cast(device_active_points_count.data()),
                                                  N);
        cudaDeviceSynchronize();
        CudaTest((char *)"build_tree_check_active_points Kernel failed!");
        check_active_points_cron.stop();

        check_points_side_cron.start();
        build_tree_check_points_side<<<NB,NT>>>(thrust::raw_pointer_cast(device_tree[depth-1]->data()),
                                   thrust::raw_pointer_cast(device_tree_parents[depth-1]->data()),
                                   thrust::raw_pointer_cast(device_tree_children[depth-1]->data()),
                                   thrust::raw_pointer_cast(device_points_parent.data()),
                                   thrust::raw_pointer_cast(device_points_depth.data()),
                                   thrust::raw_pointer_cast(device_is_right_child.data()),
                                   thrust::raw_pointer_cast(device_is_leaf[depth-1]->data()),
                                   thrust::raw_pointer_cast(device_child_count[depth-1]->data()),
                                   thrust::raw_pointer_cast(device_points.data()),
                                   thrust::raw_pointer_cast(device_actual_depth.data()),
                                   thrust::raw_pointer_cast(device_tree_count.data()),
                                   thrust::raw_pointer_cast(device_depth_level_count.data()),
                                   thrust::raw_pointer_cast(device_accumulated_child_count[depth-1]->data()),
                                   thrust::raw_pointer_cast(device_count_points_on_leaves[depth-1]->data()),
                                   thrust::raw_pointer_cast(device_sample_candidate_points.data()),
                                   thrust::raw_pointer_cast(device_points_id_on_sample.data()),
                                   thrust::raw_pointer_cast(device_active_points.data()),
                                   thrust::raw_pointer_cast(device_active_points_count.data()),
                                   thrust::raw_pointer_cast(device_count_new_nodes.data()),
                                   N, D, RANDOM_SEED);
        cudaDeviceSynchronize();
        CudaTest((char *)"build_tree_check_points_side Kernel failed!");
        check_points_side_cron.stop();
        
        tree_count_cron.start();
        thrust::fill(thrust::device, device_count_new_nodes.begin(), device_count_new_nodes.begin()+1, 0);
        cudaDeviceSynchronize();
        
        // thrust::plus<int> binary_op;
        // // compute sum on the device
        // int sum_test = thrust::reduce(device_is_leaf[depth-1]->begin(), device_is_leaf[depth-1]->end(), 0, binary_op);
        // std::cout << sum_test << " "<< device_is_leaf[depth-1]->size() << " <<<<#####" << std::endl;

        build_tree_count_new_nodes<<<NB,NT>>>(thrust::raw_pointer_cast(device_tree[depth-1]->data()),
                                              thrust::raw_pointer_cast(device_tree_parents[depth-1]->data()),
                                              thrust::raw_pointer_cast(device_tree_children[depth-1]->data()),
                                              thrust::raw_pointer_cast(device_points_parent.data()),
                                              thrust::raw_pointer_cast(device_is_right_child.data()),
                                              thrust::raw_pointer_cast(device_is_leaf[depth-1]->data()),
                                              thrust::raw_pointer_cast(device_count_points_on_leaves[depth-1]->data()),
                                              thrust::raw_pointer_cast(device_child_count[depth-1]->data()),
                                              thrust::raw_pointer_cast(device_points.data()),
                                              thrust::raw_pointer_cast(device_actual_depth.data()),
                                              thrust::raw_pointer_cast(device_tree_count.data()),
                                              thrust::raw_pointer_cast(device_depth_level_count.data()),
                                              thrust::raw_pointer_cast(device_count_new_nodes.data()),
                                              N, D, MIN_TREE_CHILD, MAX_TREE_CHILD);
        cudaDeviceSynchronize();
        CudaTest((char *)"build_tree_count_new_nodes Kernel failed!");
        thrust::copy(device_count_new_nodes.begin(), device_count_new_nodes.begin()+1, &count_new_nodes);
        cudaDeviceSynchronize();
        tree_count_cron.stop();
        
        if(count_new_nodes > 0){
            dynamic_memory_allocation_cron.start();
            count_total_nodes+=count_new_nodes;

            if(free_index){
                device_tree[depth-1]->clear();
                device_tree[depth-1]->shrink_to_fit();
            }
            
            // device_tree[depth] = new thrust::device_vector<RSFK_typepoints>((((D+19)+1)*count_new_nodes));
            device_tree[depth] = new thrust::device_vector<RSFK_typepoints>(((D+1)*count_new_nodes));
            // device_random_directions[depth] = new thrust::device_vector<RSFK_typepoints>((2*D*count_new_nodes));
            device_tree_parents[depth] = new thrust::device_vector<int>(count_new_nodes,-1);
            device_tree_children[depth] = new thrust::device_vector<int>(2*count_new_nodes,-1);
            device_is_leaf[depth] = new thrust::device_vector<bool>(count_new_nodes, true);
            device_child_count[depth] = new thrust::device_vector<int>(count_new_nodes, 0);
            device_accumulated_child_count[depth] = new thrust::device_vector<int>(2*count_new_nodes, 0);
            device_count_points_on_leaves[depth] = new thrust::device_vector<int>(2*count_new_nodes, 0);
            cudaDeviceSynchronize();
            
            build_tree_accumulate_child_count<<<1,1>>>(thrust::raw_pointer_cast(device_depth_level_count.data()),
                                                       thrust::raw_pointer_cast(device_count_new_nodes.data()),
                                                       thrust::raw_pointer_cast(device_count_points_on_leaves[depth-1]->data()),
                                                       thrust::raw_pointer_cast(device_accumulated_child_count[depth-1]->data()),
                                                       thrust::raw_pointer_cast(device_actual_depth.data()));
            cudaDeviceSynchronize();
            CudaTest((char *)"build_tree_accumulate_child_count Kernel failed!");
            dynamic_memory_allocation_cron.stop();

            
            organize_sample_candidate_cron.start();
            build_tree_organize_sample_candidates<<<NB,NT>>>(thrust::raw_pointer_cast(device_points_parent.data()),
                                                             thrust::raw_pointer_cast(device_points_depth.data()),
                                                             thrust::raw_pointer_cast(device_is_leaf[depth-1]->data()),
                                                             thrust::raw_pointer_cast(device_is_right_child.data()),
                                                             thrust::raw_pointer_cast(device_count_new_nodes.data()),
                                                             thrust::raw_pointer_cast(device_count_points_on_leaves[depth-1]->data()),
                                                             thrust::raw_pointer_cast(device_accumulated_child_count[depth-1]->data()),
                                                             thrust::raw_pointer_cast(device_sample_candidate_points.data()),
                                                             thrust::raw_pointer_cast(device_points_id_on_sample.data()),
                                                             thrust::raw_pointer_cast(device_actual_depth.data()),
                                                             N);
            cudaDeviceSynchronize();
            CudaTest((char *)"build_tree_organize_sample_candidates Kernel failed!");
            organize_sample_candidate_cron.stop();

            create_nodes_cron.start();
            // init_random_directions<<<NB,NT>>>(thrust::raw_pointer_cast(device_random_directions[depth]->data()),
            //                                   count_new_nodes*D,
            //                                   RANDOM_SEED);
            
            // thrust::fill(thrust::device, device_min_random_proj_values.begin(), device_min_random_proj_values.end(), FLT_MAX);
            // thrust::fill(thrust::device, device_max_random_proj_values.begin(), device_max_random_proj_values.end(), FLT_MIN);
        
            build_tree_create_nodes<<<NB,NT>>>(thrust::raw_pointer_cast(device_tree[depth]->data()), // new depth is created
                                               thrust::raw_pointer_cast(device_tree_parents[depth]->data()), // new depth is created
                                               thrust::raw_pointer_cast(device_tree_children[depth-1]->data()),
                                               thrust::raw_pointer_cast(device_points_parent.data()),
                                               thrust::raw_pointer_cast(device_points_depth.data()),
                                               thrust::raw_pointer_cast(device_is_right_child.data()),
                                               thrust::raw_pointer_cast(device_is_leaf[depth-1]->data()),
                                               thrust::raw_pointer_cast(device_is_leaf[depth]->data()), // new depth is created
                                               thrust::raw_pointer_cast(device_child_count[depth-1]->data()),
                                               thrust::raw_pointer_cast(device_points.data()),
                                               thrust::raw_pointer_cast(device_actual_depth.data()),
                                               thrust::raw_pointer_cast(device_tree_count.data()),
                                               thrust::raw_pointer_cast(device_depth_level_count.data()),
                                               thrust::raw_pointer_cast(device_count_new_nodes.data()),
                                               thrust::raw_pointer_cast(device_accumulated_nodes_count.data()),
                                               thrust::raw_pointer_cast(device_accumulated_child_count[depth-1]->data()),
                                               thrust::raw_pointer_cast(device_count_points_on_leaves[depth-1]->data()),
                                               thrust::raw_pointer_cast(device_sample_candidate_points.data()),
                                               N, D, MIN_TREE_CHILD, MAX_TREE_CHILD, RANDOM_SEED);
            cudaDeviceSynchronize();
            CudaTest((char *)"build_tree_create_nodes Kernel failed!");
            create_nodes_cron.stop();


            update_parents_cron.start();
            build_tree_update_parents<<<NB,NT>>>(thrust::raw_pointer_cast(device_tree[depth]->data()),
                                                thrust::raw_pointer_cast(device_tree_parents[depth-1]->data()),
                                                thrust::raw_pointer_cast(device_tree_children[depth-1]->data()),
                                                thrust::raw_pointer_cast(device_points_parent.data()),
                                                thrust::raw_pointer_cast(device_points_depth.data()),
                                                thrust::raw_pointer_cast(device_is_right_child.data()),
                                                thrust::raw_pointer_cast(device_is_leaf[depth-1]->data()),
                                                thrust::raw_pointer_cast(device_is_leaf[depth]->data()),
                                                thrust::raw_pointer_cast(device_child_count[depth-1]->data()),
                                                thrust::raw_pointer_cast(device_child_count[depth]->data()), // new depth is created
                                                thrust::raw_pointer_cast(device_points.data()),
                                                thrust::raw_pointer_cast(device_actual_depth.data()),
                                                thrust::raw_pointer_cast(device_tree_count.data()),
                                                thrust::raw_pointer_cast(device_depth_level_count.data()),
                                                thrust::raw_pointer_cast(device_count_new_nodes.data()),
                                                N, D, MIN_TREE_CHILD, MAX_TREE_CHILD);
            cudaDeviceSynchronize();
            CudaTest((char *)"build_tree_update_parents Kernel failed!");
            build_tree_post_update_parents<<<NB,NT>>>(thrust::raw_pointer_cast(device_tree[depth]->data()),
                                                    thrust::raw_pointer_cast(device_tree_parents[depth]->data()), // new depth is created
                                                    thrust::raw_pointer_cast(device_tree_children[depth-1]->data()),
                                                    thrust::raw_pointer_cast(device_points_parent.data()),
                                                    thrust::raw_pointer_cast(device_points_depth.data()),
                                                    thrust::raw_pointer_cast(device_is_right_child.data()),
                                                    thrust::raw_pointer_cast(device_is_leaf[depth-1]->data()),
                                                    thrust::raw_pointer_cast(device_is_leaf[depth]->data()),
                                                    thrust::raw_pointer_cast(device_child_count[depth-1]->data()),
                                                    thrust::raw_pointer_cast(device_child_count[depth]->data()), // new depth is created
                                                    thrust::raw_pointer_cast(device_points.data()),
                                                    thrust::raw_pointer_cast(device_actual_depth.data()),
                                                    thrust::raw_pointer_cast(device_tree_count.data()),
                                                    thrust::raw_pointer_cast(device_depth_level_count.data()),
                                                    thrust::raw_pointer_cast(device_count_new_nodes.data()),
                                                    #if RSFK_COMPILE_TYPE == RSFK_DEBUG
                                                        thrust::raw_pointer_cast(device_count_undo_leaf.data()),
                                                    #endif
                                                    N, D, MIN_TREE_CHILD, MAX_TREE_CHILD);
            cudaDeviceSynchronize();
            CudaTest((char *)"build_tree_post_update_parents Kernel failed!");
            
            
            build_tree_utils<<<1,1>>>(thrust::raw_pointer_cast(device_actual_depth.data()),
                                      thrust::raw_pointer_cast(device_depth_level_count.data()),
                                      thrust::raw_pointer_cast(device_count_new_nodes.data()),
                                      thrust::raw_pointer_cast(device_tree_count.data()),
                                      thrust::raw_pointer_cast(device_accumulated_nodes_count.data()),
                                      thrust::raw_pointer_cast(device_child_count[depth]->data()), // new depth is created
                                      thrust::raw_pointer_cast(device_accumulated_child_count[depth]->data()), // new depth is created
                                      thrust::raw_pointer_cast(device_active_points_count.data())
                                      );
            cudaDeviceSynchronize();
            CudaTest((char *)"build_tree_utils Kernel failed!");
            update_parents_cron.stop();
        }

        if(VERBOSE >= 2){
            std::cout << "\e[ABuilding Tree Depth: " << depth+1 << "/" << MAX_DEPTH;
            std::cout << " | Total nodes: " << count_total_nodes << std::endl;
        }

        if(count_new_nodes == 0){
            if(VERBOSE >= 1) std::cout << "Early stop: 0 new nodes created." << std::endl;
            break;
        }
    }
    if(free_index){
        device_tree[depth-1]->clear();
        device_tree[depth-1]->shrink_to_fit();
    }
    
    reached_max_depth = depth;
    build_tree_fix<<<1,1>>>(thrust::raw_pointer_cast(device_depth_level_count.data()),
                            thrust::raw_pointer_cast(device_tree_count.data()),
                            thrust::raw_pointer_cast(device_accumulated_nodes_count.data()),
                            reached_max_depth);
    cudaDeviceSynchronize();
    CudaTest((char *)"build_tree_fix Kernel failed!");
    
    total_tree_build_cron.stop();
    
    Cron cron_classify_points;
    cron_classify_points.start();

    thrust::device_vector<int> device_max_leaf_size(1,0);
    thrust::device_vector<int> device_min_leaf_size(1,INT_MAX);
    thrust::device_vector<int> device_total_leaves(1,0);
    
    for(depth=0; depth < reached_max_depth; ++depth){
        build_tree_max_leaf_size<<<NB,NT>>>(thrust::raw_pointer_cast(device_max_leaf_size.data()),
                                            thrust::raw_pointer_cast(device_min_leaf_size.data()),
                                            thrust::raw_pointer_cast(device_total_leaves.data()),
                                            thrust::raw_pointer_cast(device_is_leaf[depth]->data()),
                                            thrust::raw_pointer_cast(device_child_count[depth]->data()),
                                            thrust::raw_pointer_cast(device_tree_count.data()),
                                            thrust::raw_pointer_cast(device_depth_level_count.data()),
                                            depth);
        cudaDeviceSynchronize();
        CudaTest((char *)"build_tree_max_leaf_size Kernel failed!");
    }

    
    
    int max_child, total_leaves, min_child;
    thrust::copy(device_max_leaf_size.begin(), device_max_leaf_size.begin()+1, &max_child);
    thrust::copy(device_min_leaf_size.begin(), device_min_leaf_size.begin()+1, &min_child);
    thrust::copy(device_total_leaves.begin(), device_total_leaves.begin()+1, &total_leaves);
    cudaDeviceSynchronize();
    
    thrust::copy(device_tree_count.begin(), device_tree_count.begin()+1, &MAX_NODES);
    
    #if RSFK_COMPILE_TYPE == RSFK_DEBUG
        int count_undo_leaf;
        thrust::copy(device_count_undo_leaf.begin(), device_count_undo_leaf.begin()+1, &count_undo_leaf);
        cudaDeviceSynchronize();
        std::cout << "device_count_undo_leaf: " << count_undo_leaf << std::endl;

        thrust::device_vector<int> device_hist_leaves_size(max_child-min_child+1,0);
        for(depth=0; depth < reached_max_depth; ++depth){
            debug_count_hist_leaf_size<<<NB,NT>>>(thrust::raw_pointer_cast(device_hist_leaves_size.data()),
                                                  thrust::raw_pointer_cast(device_max_leaf_size.data()),
                                                  thrust::raw_pointer_cast(device_min_leaf_size.data()),
                                                  thrust::raw_pointer_cast(device_total_leaves.data()),
                                                  thrust::raw_pointer_cast(device_is_leaf[depth]->data()),
                                                  thrust::raw_pointer_cast(device_child_count[depth]->data()),
                                                  thrust::raw_pointer_cast(device_tree_count.data()),
                                                  thrust::raw_pointer_cast(device_depth_level_count.data()),
                                                  depth);
            cudaDeviceSynchronize();
            CudaTest((char *)"debug_count_hist_leaf_size Kernel failed!");
        }
        std::vector<int> hist_leaves_size;
        hist_leaves_size.resize(max_child-min_child+1);
        thrust::copy(device_hist_leaves_size.begin(),
                     device_hist_leaves_size.begin() + max_child-min_child+1,
                     hist_leaves_size.begin());
        device_hist_leaves_size.clear();
        device_hist_leaves_size.shrink_to_fit();
        std::cout << "Leaves count histogram (size | count):" << std::endl;
        for(int i=min_child; i <= max_child; ++i){
            std::cout << i << "\t" << hist_leaves_size[i-min_child] << std::endl;
        }
        std::cout << std::endl;
    #endif
    
    if(VERBOSE >= 1){
        std::cout << "Tree depth: " << reached_max_depth << std::endl;
        std::cout << "Max child count: " << max_child << std::endl;
        std::cout << "Min child count: " << min_child << std::endl;
        std::cout << "Total leaves: " << total_leaves << std::endl;
    }

    thrust::device_vector<int> device_leaf_idx_to_node_idx(total_leaves, -1);
    thrust::device_vector<int> device_node_idx_to_leaf_idx(MAX_NODES, -1);
    thrust::device_vector<int> device_nodes_buckets(total_leaves*max_child, -1);
    thrust::device_vector<int> device_bucket_sizes(total_leaves, 0);

    thrust::fill(thrust::device, device_total_leaves.begin(), device_total_leaves.end(), 0);
    cudaDeviceSynchronize();

    for(depth=0; depth < reached_max_depth; ++depth){
        build_tree_set_leaves_idx<<<NB,NT>>>(thrust::raw_pointer_cast(device_leaf_idx_to_node_idx.data()),
                                            thrust::raw_pointer_cast(device_node_idx_to_leaf_idx.data()),
                                            thrust::raw_pointer_cast(device_total_leaves.data()),
                                            thrust::raw_pointer_cast(device_is_leaf[depth]->data()),
                                            thrust::raw_pointer_cast(device_depth_level_count.data()),
                                            thrust::raw_pointer_cast(device_accumulated_nodes_count.data()),
                                            depth);
        cudaDeviceSynchronize();
        CudaTest((char *)"build_tree_set_leaves_idx Kernel failed!");
    }
    
    
    build_tree_bucket_points<<<NB,NT>>>(thrust::raw_pointer_cast(device_points_parent.data()),
                                        thrust::raw_pointer_cast(device_points_depth.data()),
                                        thrust::raw_pointer_cast(device_accumulated_nodes_count.data()),
                                        thrust::raw_pointer_cast(device_node_idx_to_leaf_idx.data()),
                                        thrust::raw_pointer_cast(device_nodes_buckets.data()),
                                        thrust::raw_pointer_cast(device_bucket_sizes.data()),
                                        N, max_child, total_leaves);
    cudaDeviceSynchronize();
    CudaTest((char *)"build_tree_bucket_points Kernel failed!");
    cron_classify_points.stop();
    end_tree_cron.start();

    if(free_index){
        device_leaf_idx_to_node_idx.clear();
        device_leaf_idx_to_node_idx.shrink_to_fit();
        device_node_idx_to_leaf_idx.clear();
        device_node_idx_to_leaf_idx.shrink_to_fit();
        device_points_parent.clear();
        device_points_parent.shrink_to_fit();
        device_points_depth.clear();
        device_points_depth.shrink_to_fit();
        device_is_right_child.clear();
        device_is_right_child.shrink_to_fit();
        device_sample_candidate_points.clear();
        device_sample_candidate_points.shrink_to_fit();
        device_points_id_on_sample.clear();
        device_points_id_on_sample.shrink_to_fit();

        device_depth_level_count.clear();
        device_depth_level_count.shrink_to_fit();
        device_accumulated_nodes_count.clear();
        device_accumulated_nodes_count.shrink_to_fit();
        device_tree_count.clear();
        device_tree_count.shrink_to_fit();
        device_count_new_nodes.clear();
        device_count_new_nodes.shrink_to_fit();
        device_actual_depth.clear();
        device_actual_depth.shrink_to_fit();
        
        device_max_leaf_size.clear();
        device_max_leaf_size.shrink_to_fit();
        device_min_leaf_size.clear();
        device_min_leaf_size.shrink_to_fit();
        device_total_leaves.clear();
        device_total_leaves.shrink_to_fit();

        for(depth=0; depth < reached_max_depth; ++depth){
            // Random Projection Forest
            // device_random_directions[depth]->clear();
            // device_random_directions[depth]->shrink_to_fit();
            // device_min_random_proj_values[depth]->clear();
            // device_min_random_proj_values[depth]->shrink_to_fit();
            // device_max_random_proj_values[depth]->clear();
            // device_max_random_proj_values[depth]->shrink_to_fit();
            device_tree_parents[depth]->clear();
            device_tree_parents[depth]->shrink_to_fit();
            device_tree_children[depth]->clear();
            device_tree_children[depth]->shrink_to_fit();
            device_is_leaf[depth]->clear();
            device_is_leaf[depth]->shrink_to_fit();
            device_child_count[depth]->clear();
            device_child_count[depth]->shrink_to_fit();
            device_accumulated_child_count[depth]->clear();
            device_accumulated_child_count[depth]->shrink_to_fit();
            device_count_points_on_leaves[depth]->clear();
            device_count_points_on_leaves[depth]->shrink_to_fit();
        }
    }
    else{
        rsfkindextree->device_tree = device_tree;
        rsfkindextree->device_tree_parents = device_tree_parents;
        rsfkindextree->device_tree_children = device_tree_children;
        rsfkindextree->device_is_leaf = device_is_leaf;
        rsfkindextree->device_child_count = device_child_count;
        rsfkindextree->device_accumulated_child_count = device_accumulated_child_count;
        rsfkindextree->device_count_points_on_leaves = device_count_points_on_leaves;

        rsfkindextree->device_points_parent = &device_points_parent;
        rsfkindextree->device_points_depth = &device_points_depth;
        rsfkindextree->device_is_right_child = &device_is_right_child;
        rsfkindextree->device_sample_candidate_points = &device_sample_candidate_points;

        rsfkindextree->device_points_id_on_sample = &device_points_id_on_sample;
        rsfkindextree->device_active_points = &device_active_points;

        rsfkindextree->device_depth_level_count = &device_depth_level_count;
        rsfkindextree->device_accumulated_nodes_count = &device_accumulated_nodes_count;
        rsfkindextree->device_tree_count = &device_tree_count;
        rsfkindextree->device_count_new_nodes = &device_count_new_nodes;
        rsfkindextree->device_actual_depth = &device_actual_depth;
        rsfkindextree->device_active_points_count = &device_active_points_count;

        rsfkindextree->device_leaf_idx_to_node_idx = &device_leaf_idx_to_node_idx;
        rsfkindextree->device_node_idx_to_leaf_idx = &device_node_idx_to_leaf_idx;
        rsfkindextree->device_nodes_buckets = &device_nodes_buckets;
        rsfkindextree->device_bucket_sizes = &device_bucket_sizes;


        rsfkindextree->total_leaves = total_leaves;
        rsfkindextree->max_child = max_child;
    }

    // do 
    // {
    // std::cout << '\n' << "Press a key to continue...";
    // } while (std::cin.get() != '\n');


    end_tree_cron.stop();
    // Report total time of each step
    if(VERBOSE >= 1){
        printf("Init tree takes %lf seconds\n", init_tree_cron.t_total/1000);
        printf("Total tree building takes %lf seconds\n", total_tree_build_cron.t_total/1000);
        printf("\tCheck active points Kernel takes %lf seconds\n", check_active_points_cron.t_total/1000);
        printf("\tCheck points side Kernel takes %lf seconds\n", check_points_side_cron.t_total/1000);
        printf("\tCount new nodes Kernel takes %lf seconds\n", tree_count_cron.t_total/1000);
        printf("\tDynamic memory allocation takes %lf seconds\n", dynamic_memory_allocation_cron.t_total/1000);
        printf("\tPreprocessing split points Kernel takes %lf seconds\n", organize_sample_candidate_cron.t_total/1000);
        printf("\tCreate nodes Kernel takes %lf seconds\n", create_nodes_cron.t_total/1000);
        printf("\tUpdate parents Kernel takes %lf seconds\n", update_parents_cron.t_total/1000);
        printf("Bucket creation Kernel takes %lf seconds\n", cron_classify_points.t_total/1000);
        printf("End tree takes %lf seconds\n", end_tree_cron.t_total/1000);
    }
    
    forest_log.update_log(
        reached_max_depth,
        max_child,
        min_child,
        total_leaves,
        (float)init_tree_cron.t_total/1000,
        (float)total_tree_build_cron.t_total/1000,
        (float)check_active_points_cron.t_total/1000,
        (float)check_points_side_cron.t_total/1000,
        (float)tree_count_cron.t_total/1000,
        (float)dynamic_memory_allocation_cron.t_total/1000,
        (float)organize_sample_candidate_cron.t_total/1000,
        (float)create_nodes_cron.t_total/1000,
        (float)update_parents_cron.t_total/1000,
        (float)cron_classify_points.t_total/1000,
        (float)end_tree_cron.t_total/1000
    );

    TreeInfo tinfo = TreeInfo(total_leaves, max_child,
                              device_nodes_buckets, device_bucket_sizes);
    
    
    if(!free_index){
        rsfkindextree->reached_max_depth = reached_max_depth;
    }

    return tinfo;
}

/*
__global__
void create_point_to_anchor(int* point_to_anchor,
                            int* points_buckets,
                            int* bucket_sizes,
                            int total_buckets,
                            int max_bucket_size)
{
    int tid = blockDim.x*blockIdx.x+threadIdx.x;
    int p, i;
    // Some threads try to process the padding (-1) values
    // TODO: Optmize and ensure a coalesced access pattern 
    for(i = tid; i < total_buckets*max_bucket_size; i+=blockDim.x*gridDim.x){
        __syncthreads();
        p = points_buckets[i];
        if(p == -1) continue;
        point_to_anchor[p] = i/max_bucket_size;
    }
}

__global__
void symmetric_knn_graph_edge_count(int* knn_indices,
                                    int* node_edge_count,
                                    int N,
                                    int num_neighbors)
{
    int tid = blockDim.x*blockIdx.x+threadIdx.x;
    int p, pn;
    bool is_symetric;
    for(p = tid; p < N; p+=blockDim.x*gridDim.x){
        for(int i=0; i < num_neighbors; ++i){
            pn = knn_indices[p*num_neighbors + i];
            atomicAdd(&node_edge_count[p], 1);
            is_symetric = false;
            for(int j = 0; j < num_neighbors; ++j){
                is_symetric |= knn_indices[pn*num_neighbors + j] == p;
            }

            // Force the bidirecional edge if it doesnt exists
            if(!is_symetric) atomicAdd(&node_edge_count[pn], 1);
        }
    }
}


// TODO: Implement cumulative sum
__global__
void symmetric_knn_graph_offset(int* node_edge_count,
                                int* nodes_offset,
                                int N)
{
    int tid = blockDim.x*blockIdx.x+threadIdx.x;
    int p, pn;
    int sum = 0;
    if(threadIdx.x == 0){
        for(p = 0; p < N; ++p){
            nodes_offset[p] = sum;
            sum+= node_edge_count[p];
        }
    }
}

__global__
void symmetric_knn_graph_add_edges(int* knn_indices,
                                   int* knn_sym_indices,
                                   int* node_edge_count,
                                   int* nodes_offset,
                                   int N,
                                   int num_neighbors)
{
    int tid = blockDim.x*blockIdx.x+threadIdx.x;
    int p, pn, edge_idx;
    bool is_symetric;
    for(p = tid; p < N; p+=blockDim.x*gridDim.x){
        for(int i=0; i < num_neighbors; ++i){
            pn = knn_indices[p*num_neighbors + i];
            edge_idx = atomicAdd(&node_edge_count[p], 1);
            knn_sym_indices[nodes_offset[p]+edge_idx] = pn;

            is_symetric = false;
            for(int j = 0; j < num_neighbors; ++j){
                is_symetric |= knn_indices[pn*num_neighbors + j] == p;
            }
            // Force the bidirecional edge if it doesnt exists
            if(!is_symetric){
                edge_idx = atomicAdd(&node_edge_count[pn], 1);
                knn_sym_indices[nodes_offset[pn]+edge_idx] = p;
            }
        }
    }
}



struct FunctionalSqrt {
    __host__ __device__ float operator()(const float &x) const {
        return pow(x, 0.5);
    }
};



int RSFK::spectral_clustering_with_knngraph(int* result, int num_neighbors,
                                            int N, int D, int VERBOSE,
                                            int K, int n_eig_vects,
                                            bool free_knn_indices=true,
                                            std::string run_name="tree")
{
    
    if(knn_indices == nullptr){
        printf("%s:%d: Error: spectral_clustering_with_knngraph called"
               " without a valid k-nn graph\n");
        return 1;
    }

    int devUsed = 0;
    cudaSetDevice(devUsed);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, devUsed);
    const int NT = deviceProp.maxThreadsPerBlock;
    const int NB = deviceProp.multiProcessorCount*(deviceProp.maxThreadsPerMultiProcessor/deviceProp.maxThreadsPerBlock);
    

    thrust::device_vector<int> device_knn_indices(
        knn_indices, knn_indices+N*num_neighbors);
    
    thrust::device_vector<int> device_node_edge_count(N,0);
    thrust::device_vector<int> device_node_edge_count_prefixsum(N);

    // TODO: Ignore redundant edges
    symmetric_knn_graph_edge_count<<<NB,NT>>>(
        thrust::raw_pointer_cast(device_knn_indices.data()),
        thrust::raw_pointer_cast(device_node_edge_count.data()),
        N,
        num_neighbors);
    cudaDeviceSynchronize();
    CudaTest((char *)"symmetric_knn_graph_edge_count Kernel failed!");

    thrust::exclusive_scan(thrust::device,
                           device_node_edge_count.begin(),
                           device_node_edge_count.end(),
                           device_node_edge_count_prefixsum.begin());

    int n_edges = thrust::reduce(device_node_edge_count.begin(),
                                 device_node_edge_count.end(), 
                                 0, thrust::plus<int>());
    thrust::fill(device_node_edge_count.begin(),
                 device_node_edge_count.end(),
                 0);
    
    thrust::device_vector<int> device_knn_sym_indices(n_edges);
    
    symmetric_knn_graph_add_edges<<<NB,NT>>>(
        thrust::raw_pointer_cast(device_knn_indices.data()),
        thrust::raw_pointer_cast(device_knn_sym_indices.data()),
        thrust::raw_pointer_cast(device_node_edge_count.data()),
        thrust::raw_pointer_cast(device_node_edge_count_prefixsum.data()),
        N,
        num_neighbors);
    cudaDeviceSynchronize();
    CudaTest((char *)"symmetric_knn_graph_add_edges Kernel failed!");

    if(free_knn_indices){
        free(knn_indices);
        // free(knn_sqr_distances);
        knn_indices = nullptr;
        // knn_sqr_distances = nullptr;
    }
    int* knn_indices_sym = new int[n_edges];
    int* source_offsets = new int[N];
    thrust::copy(device_knn_sym_indices.begin(),
                 device_knn_sym_indices.end(),
                 knn_indices_sym);
    thrust::copy(device_node_edge_count_prefixsum.begin(),
                 device_node_edge_count_prefixsum.end(),
                 source_offsets);

    if(free_knn_indices){
        knn_indices = knn_indices_sym;
    }

    

    nvgraphStatus_t status;

    Cron cluster_forest_cron;
    cluster_forest_cron.start();

    // n_edges = N*num_neighbors;
    int n_vertices = N;
    if(VERBOSE >= 1){
        printf("Creating graph with %d vertices and %d edges\n",
               n_vertices, n_edges);
    }

    nvgraphCSRTopology32I_st nvgraph_top;
    nvgraph_top.nvertices = n_vertices;
    nvgraph_top.nedges = n_edges;
    

    // CREATE GRAPH OFFSET indexes

    // // Assumes that the knn_indices are precomputed
    
    // nvgraph_top.source_offsets = new int[n_vertices];
    // for(int i=0; i < N; ++i){
    //     // each data point contains an edge to a cluster for each tree
    //     nvgraph_top.source_offsets[i] = i*num_neighbors;
    // }
    // // nvgraph_top.destination_indices = knn_indices;
    
    nvgraph_top.source_offsets = source_offsets;
    nvgraph_top.destination_indices = knn_indices_sym;

    // TODO: Use squared distances as edge value?
    float* edgevals = new float[n_edges];
    // float* edgevals = knn_sqr_distances;
    
    std::fill_n(edgevals, n_edges, 1.0f);

    
    // thrust::transform(knn_sqr_distances, knn_sqr_distances+n_edges,
    //                   edgevals, FunctionalSqrt());


    // float norm_val = thrust::reduce(
    //     edgevals,
    //     edgevals+n_edges,
    //     FLT_MIN, thrust::maximum<float>());
    
    // thrust::transform(edgevals, edgevals+n_edges,
    //                   thrust::make_constant_iterator(norm_val),
    //                   edgevals,
    //                   thrust::minus<float>());
    
    // norm_val = thrust::reduce(
    //     edgevals,
    //     edgevals+n_edges,
    //     FLT_MAX, thrust::minimum<float>());

    // std::cout << norm_val << std::endl; 
    // thrust::transform(edgevals,
    //                   edgevals+n_edges,
    //                   thrust::constant_iterator<float>(norm_val),
    //                   edgevals,
    //                   thrust::divides<float>());

    // thrust::transform(edgevals, edgevals+n_edges,
    //                   thrust::make_constant_iterator(1.0),
    //                   edgevals,
    //                   thrust::plus<float>());
    // thrust::transform(edgevals,
    //                   edgevals+n_edges,
    //                   thrust::constant_iterator<float>(2.0),
    //                   edgevals,
    //                   thrust::divides<float>());

    // if(VERBOSE >= 2){
    //     norm_val = thrust::reduce(
    //         edgevals,
    //         edgevals+n_edges,
    //         FLT_MAX, thrust::minimum<float>());
    //     printf("Minimum edge value: %f\n", norm_val);
    //     norm_val = thrust::reduce(
    //         edgevals,
    //         edgevals+n_edges,
    //         FLT_MIN, thrust::maximum<float>());
    //     printf("Maximum edge value: %f\n", norm_val);
    // }



    // PREPARES THE LAUNCH OF THE nvgraphSpectralClustering KERNEL
    nvgraphHandle_t nvgraph_handle;
    nvgraphGraphDescr_t descrG;
    cudaDataType_t edge_dimT = CUDA_R_32F;

    status = nvgraphCreate(&nvgraph_handle);
    cudaDeviceSynchronize();
    if(status != NVGRAPH_STATUS_SUCCESS){
        printf("%s:%d :\t nvgraphCreate fail!\n",
                __FILE__, __LINE__);
    }
    status = nvgraphCreateGraphDescr(nvgraph_handle, &descrG);
    cudaDeviceSynchronize();
    if(status != NVGRAPH_STATUS_SUCCESS){
        printf("%s:%d :\t nvgraphCreateGraphDescr fail!\n",
                __FILE__, __LINE__);
    }
    status = nvgraphSetGraphStructure(nvgraph_handle,
                                      descrG,
                                      (void*)&nvgraph_top,
                                      NVGRAPH_CSR_32);
    cudaDeviceSynchronize();
    if(status != NVGRAPH_STATUS_SUCCESS){
        printf("%s:%d :\t nvgraphSetGraphStructure fail!\n",
                __FILE__, __LINE__);
    }
    status = nvgraphAllocateEdgeData(nvgraph_handle, descrG, 1, &edge_dimT);
    cudaDeviceSynchronize();
    if(status != NVGRAPH_STATUS_SUCCESS){
        printf("%s:%d :\t nvgraphAllocateEdgeData fail!\n",
                __FILE__, __LINE__);
    }

    status = nvgraphSetEdgeData(nvgraph_handle, descrG, (void*)edgevals, 0);
    cudaDeviceSynchronize();
    if(status != NVGRAPH_STATUS_SUCCESS){
        printf("%s:%d :\t nvgraphSetEdgeData fail!\n",
                __FILE__, __LINE__);
    }

    SpectralClusteringParameter specParam;
    
    specParam.n_clusters = K;
    specParam.n_eig_vects = n_eig_vects;
    
    // NVGRAPH_MODULARITY_MAXIMIZATION : maximize modularity with Lanczos solver.
    // NVGRAPH_BALANCED_CUT_LANCZOS : minimize balanced cut with Lanczos solver.
    // NVGRAPH_BALANCED_CUT_LOBPCG: minimize balanced cut with LOPCG solver. 
    
    specParam.algorithm = NVGRAPH_MODULARITY_MAXIMIZATION; 
    specParam.evs_tolerance = 0.0f; // default value
    specParam.evs_max_iter = 0; // default value 
    // specParam.evs_tolerance = 0.01f; // default value
    // specParam.evs_max_iter = 10000; // default value 

    specParam.kmean_tolerance = 0.0f; // default value
    specParam.kmean_max_iter = 0; // default value 
    // specParam.kmean_tolerance = 0.0001f; // default value
    // specParam.kmean_max_iter = 3000; // default value 


    int weight_index = 0;
    float* result_eigvals;
    float* result_eigvecs;

    int* result_clustering = new int[nvgraph_top.nvertices];
    int attempts = 0;
    do{
        if(attempts > 0){
            delete [] result_eigvals;
            delete [] result_eigvecs;
        }
        result_eigvals  = new float[n_eig_vects];
        result_eigvecs  = new float[n_eig_vects*nvgraph_top.nvertices];
        
        if(VERBOSE >= 2){
            printf("Running nvgraphSpectralClustering with %d clusters"
                " and %d eigenvectors\n",
                specParam.n_clusters, specParam.n_eig_vects);
        }
        status = nvgraphSpectralClustering(nvgraph_handle,
                                        descrG,
                                        weight_index, // const size_t weight_index
                                        &specParam,
                                        result_clustering,
                                        result_eigvals,
                                        result_eigvecs);
        attempts++;
        specParam.n_clusters++;
        specParam.n_eig_vects++;

        if(status == NVGRAPH_STATUS_NOT_CONVERGED){
            printf("%s:%d :\t nvgraphSpectralClustering error:"
                   " NVGRAPH_STATUS_NOT_CONVERGED!\n",
                __FILE__, __LINE__);
        }
        if(status == NVGRAPH_STATUS_INVALID_VALUE){
            printf("%s:%d :\t nvgraphSpectralClustering error:"
                   " NVGRAPH_STATUS_INVALID_VALUE!\n",
                __FILE__, __LINE__);
        }
        if(status == NVGRAPH_STATUS_TYPE_NOT_SUPPORTED){
            printf("%s:%d :\t nvgraphSpectralClustering error:"
                   " NVGRAPH_STATUS_TYPE_NOT_SUPPORTED!\n",
                __FILE__, __LINE__);
        }
        if(status == NVGRAPH_STATUS_GRAPH_TYPE_NOT_SUPPORTED){
            printf("%s:%d :\t nvgraphSpectralClustering error:"
                   " NVGRAPH_STATUS_GRAPH_TYPE_NOT_SUPPORTED!\n",
                __FILE__, __LINE__);
        }

    } while(status != NVGRAPH_STATUS_SUCCESS && attempts < 10);
    cudaDeviceSynchronize();
    if(status != NVGRAPH_STATUS_SUCCESS){
        printf("%s:%d :\t nvgraphSpectralClustering fail after %d attempts!"
               " (status %d)\n",
                __FILE__, __LINE__, attempts, status);
    }

    if(VERBOSE >= 2){
        printf("Eigenvalues: [");
        for(int i=0; i < specParam.n_eig_vects; ++i){
            printf("%f", result_eigvals[i]);
            if(i+1 < specParam.n_eig_vects) printf(", ");
        }
        printf("]\n");
    }

    cluster_forest_cron.stop();
    if(VERBOSE >= 1){
        printf("Creating cluster with forest takes %lf seconds\n",
               cluster_forest_cron.t_total/1000);
    }
    

    thrust::copy(result_clustering,
                 result_clustering + N, // Ignore cluster nodes
                 result);
    cudaDeviceSynchronize();
    delete [] source_offsets;
    delete [] edgevals;
    delete [] result_clustering;
    delete [] result_eigvals;
    delete [] result_eigvecs;

    if(!free_knn_indices){
        delete [] knn_indices_sym;
    }
    return 0;
}


int RSFK::create_cluster_with_hbgf(int* result, int n_trees,
                                   int N, int D, int VERBOSE,
                                   int K, int n_eig_vects,
                                   std::string run_name="tree")
{
    int devUsed = 0;
    cudaSetDevice(devUsed);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, devUsed);
    const int NT = deviceProp.maxThreadsPerBlock;
    const int NB = deviceProp.multiProcessorCount*(deviceProp.maxThreadsPerMultiProcessor/deviceProp.maxThreadsPerBlock);
    
    nvgraphStatus_t status;

    Cron cluster_forest_cron;
    cluster_forest_cron.start();
    thrust::device_vector<RSFK_typepoints> device_points(points, points+N*D);
    
    // TreeInfo* tinfo_list = (TreeInfo*)malloc(sizeof(TreeInfo)*n_trees);
    TreeInfo* tinfo_list = new TreeInfo[n_trees];
    ForestLog forest_log = ForestLog(n_trees);

    // total_clusters
    // allocate structures to spectral clustering
    // for each tree
    //   create_point_to_anchor
    //   for each point add edges to it cluster in graph
    //   for each cluster add edge to its points (undirected graph)
    // cluster with spectral cluster and return consensus


    int n_clusters = 0;
    int max_bucket_size = 0;
    int max_bucket_array_size = 0;
    for(int i=0; i < n_trees; ++i){
        TreeInfo tinfo = create_bucket_from_sample_tree(device_points,
                                                        N, D, VERBOSE-1,
                                                        forest_log,
                                                        run_name+".png");
        tinfo_list[i] = tinfo;
        n_clusters += tinfo.total_leaves;
        max_bucket_size = max(max_bucket_size, tinfo.max_child);
        max_bucket_array_size = max(max_bucket_array_size,
                                    tinfo.total_leaves*tinfo.max_child);
        RANDOM_SEED++;
    }
    device_points.clear();
    device_points.shrink_to_fit();

    int n_edges = 2*N*n_trees;
    int n_vertices = N+n_clusters;
    if(VERBOSE >= 1){
        printf("Creating graph with %d vertices and %d edges\n",
               n_vertices, n_edges);
    }


    nvgraphCSRTopology32I_st nvgraph_top;
    nvgraph_top.nvertices = n_vertices;
    nvgraph_top.nedges = n_edges;
    // nvgraph_top.source_offsets = (int*) malloc(sizeof(int)*n_vertices);
    nvgraph_top.source_offsets = new int[n_vertices];
    // nvgraph_top.destination_indices = (int*) malloc(sizeof(int)*n_edges);
    nvgraph_top.destination_indices = new int[n_edges];


    // CREATE GRAPH TOPOLOGY OFFSET OF DATA POINTS NODES 
    for(int i=0; i < N; ++i){
        // each data point contains an edge to a cluster for each tree
        nvgraph_top.source_offsets[i] = i*n_trees;
    }

    // CREATE GRAPH TOPOLOGY OFFSET AND EDGES OF CLUSTER NODES

    // Cluster nodes in graph are placed after the nodes of data points
    int offset_sum = N*n_trees;
    int offset_cluster = 0;
    int* host_bucket_sizes = new int[max_bucket_size];
    int* host_bucket_array = new int[max_bucket_array_size];
    thrust::device_vector<int> d_nodes_buckets;
    thrust::device_vector<int> d_bucket_sizes;
    
    if(VERBOSE >= 2){
        std::cout << "Creating graph topology: offset and edges of clusters";
        std::cout << std::endl;
    }
    for(int i=0; i < n_trees; ++i){
        int total_leaves = tinfo_list[i].total_leaves;
        int max_child = tinfo_list[i].max_child;

        d_bucket_sizes = tinfo_list[i].device_bucket_sizes;
        d_nodes_buckets = tinfo_list[i].device_nodes_buckets;

        // Copy the cluster of a tree from GPU to host memory
        thrust::copy(d_bucket_sizes.begin(),
                     d_bucket_sizes.begin() + total_leaves,
                     host_bucket_sizes);
        thrust::copy(d_nodes_buckets.begin(),
                     d_nodes_buckets.begin() + total_leaves*max_child,
                     host_bucket_array);
        cudaDeviceSynchronize();

        // For each cluster of the tree
        for(int j=0; j < total_leaves; ++j){
            // Set the offset of cluster node in graph structure
            // following its size
            nvgraph_top.source_offsets[N+offset_cluster] = offset_sum;

            for(int k=0; k < host_bucket_sizes[j]; ++k){
                int point_id_on_bucket = max_child*j + k;
                // Data point Node id
                int graph_npid = host_bucket_array[point_id_on_bucket];

                // Each edge between a cluster and the related points
                // are placed sequentially in the destination_indices array
                nvgraph_top.destination_indices[offset_sum] = graph_npid;
                offset_sum++;
            }
            offset_cluster++;
        }
    }
    
    delete [] host_bucket_sizes;
    delete [] host_bucket_array;
    


    // CREATE DATA POINTS NODES EDGES IN THE GRAPH
    if(VERBOSE >= 2){
        std::cout << "Creating graph topology: offset and edges of data points";
        std::cout << std::endl;
    }

    thrust::device_vector<int> device_tree_cluster(N);
    int* host_tree_cluster = new int[N];

    offset_cluster = 0;
    for(int i=0; i < n_trees; ++i){
        int total_leaves = tinfo_list[i].total_leaves;
        int max_child = tinfo_list[i].max_child;

        d_bucket_sizes = tinfo_list[i].device_bucket_sizes;
        d_nodes_buckets = tinfo_list[i].device_nodes_buckets;

        // Get the cluster id for each point
        create_point_to_anchor<<<NB,NT>>>(
            thrust::raw_pointer_cast(device_tree_cluster.data()),
            thrust::raw_pointer_cast(d_nodes_buckets.data()),
            thrust::raw_pointer_cast(d_bucket_sizes.data()),
            total_leaves,
            max_child);
        cudaDeviceSynchronize();
        // Transfer the data to host memory
        thrust::copy(device_tree_cluster.begin(),
                     device_tree_cluster.begin() + N,
                     host_tree_cluster);
        cudaDeviceSynchronize();

        for(int j=0; j < N; ++j){
            // Data point -> Cluster edge position
            int graph_npid = j*n_trees + i;
            // Cluster node id
            int graph_ncid = N+offset_cluster+host_tree_cluster[j];

            nvgraph_top.destination_indices[graph_npid] = graph_ncid;

        }

        offset_cluster+= total_leaves;
    }
    delete [] host_tree_cluster;
    device_tree_cluster.clear();
    device_tree_cluster.shrink_to_fit();
    // PREPARES THE LAUNCH OF THE nvgraphSpectralClustering KERNEL
    nvgraphHandle_t nvgraph_handle;
    nvgraphGraphDescr_t descrG;
    cudaDataType_t edge_dimT = CUDA_R_32F;

    status = nvgraphCreate(&nvgraph_handle);
    cudaDeviceSynchronize();
    if(status != NVGRAPH_STATUS_SUCCESS){
        printf("%s:%d :\t nvgraphCreate fail!\n",
                __FILE__, __LINE__);
    }
    status = nvgraphCreateGraphDescr(nvgraph_handle, &descrG);
    cudaDeviceSynchronize();
    if(status != NVGRAPH_STATUS_SUCCESS){
        printf("%s:%d :\t nvgraphCreateGraphDescr fail!\n",
                __FILE__, __LINE__);
    }
    status = nvgraphSetGraphStructure(nvgraph_handle,
                                      descrG,
                                      (void*)&nvgraph_top,
                                      NVGRAPH_CSR_32);
    cudaDeviceSynchronize();
    if(status != NVGRAPH_STATUS_SUCCESS){
        printf("%s:%d :\t nvgraphSetGraphStructure fail!\n",
                __FILE__, __LINE__);
    }
    status = nvgraphAllocateEdgeData(nvgraph_handle, descrG, 1, &edge_dimT);
    cudaDeviceSynchronize();
    if(status != NVGRAPH_STATUS_SUCCESS){
        printf("%s:%d :\t nvgraphAllocateEdgeData fail!\n",
                __FILE__, __LINE__);
    }

    float* edgevals = new float[nvgraph_top.nedges];
    std::fill_n(edgevals, nvgraph_top.nedges, 1.0f);

    status = nvgraphSetEdgeData(nvgraph_handle, descrG, (void*)edgevals, 0);
    cudaDeviceSynchronize();
    if(status != NVGRAPH_STATUS_SUCCESS){
        printf("%s:%d :\t nvgraphSetEdgeData fail!\n",
                __FILE__, __LINE__);
    }

    SpectralClusteringParameter specParam;
    
    specParam.n_clusters = K;
    // specParam.n_clusters = tinfo_list[0].total_leaves;
    specParam.n_eig_vects = n_eig_vects;
    
    // NVGRAPH_MODULARITY_MAXIMIZATION : maximize modularity with Lanczos solver.
    // NVGRAPH_BALANCED_CUT_LANCZOS : minimize balanced cut with Lanczos solver.
    // NVGRAPH_BALANCED_CUT_LOBPCG: minimize balanced cut with LOPCG solver. 
    
    specParam.algorithm = NVGRAPH_BALANCED_CUT_LOBPCG; 
    specParam.evs_tolerance = 0.0f; // default value
    specParam.evs_max_iter = 0; // default value 
    specParam.kmean_tolerance = 0.0f; // default value
    specParam.kmean_max_iter = 0; // default value 
    // specParam.evs_tolerance = 0.00001f; // default value
    // specParam.evs_max_iter = 10000; // default value 
    // specParam.kmean_tolerance = 0.0001f; // default value
    // specParam.kmean_max_iter = 1000; // default value 


    int weight_index = 0;
    // int* result_clustering = (int*) malloc(sizeof(int)*nvgraph_top.nedges);
    // float* result_eigvals = (float*) malloc(sizeof(float)*n_eig_vects);
    // float* result_eigvecs = (float*) malloc(sizeof(float)*n_eig_vects*nvgraph_top.nedges);
    int* result_clustering = new int[nvgraph_top.nvertices];
    float* result_eigvals  = new float[n_eig_vects];
    float* result_eigvecs  = new float[n_eig_vects*nvgraph_top.nvertices];

    if(VERBOSE >= 2){
        printf("Running nvgraphSpectralClustering with %d clusters"
               " and %d eigenvectors\n", K, n_eig_vects);
    }

    
    status = nvgraphSpectralClustering(nvgraph_handle,
                                       descrG,
                                       weight_index, // const size_t weight_index
                                       &specParam,
                                       result_clustering,
                                       result_eigvals,
                                       result_eigvecs);
    cudaDeviceSynchronize();
    if(status != NVGRAPH_STATUS_SUCCESS){
        printf("%s:%d :\t nvgraphSpectralClustering fail! (status %d)\n",
                __FILE__, __LINE__, status);
        
        if(status == NVGRAPH_STATUS_NOT_CONVERGED){
            printf("%s:%d :\t nvgraphSpectralClustering error:"
                   " NVGRAPH_STATUS_NOT_CONVERGED!\n",
                __FILE__, __LINE__);
        }
        if(status == NVGRAPH_STATUS_INVALID_VALUE){
            printf("%s:%d :\t nvgraphSpectralClustering error:"
                   " NVGRAPH_STATUS_INVALID_VALUE!\n",
                __FILE__, __LINE__);
        }
        if(status == NVGRAPH_STATUS_TYPE_NOT_SUPPORTED){
            printf("%s:%d :\t nvgraphSpectralClustering error:"
                   " NVGRAPH_STATUS_TYPE_NOT_SUPPORTED!\n",
                __FILE__, __LINE__);
        }
        if(status == NVGRAPH_STATUS_GRAPH_TYPE_NOT_SUPPORTED){
            printf("%s:%d :\t nvgraphSpectralClustering error:"
                   " NVGRAPH_STATUS_GRAPH_TYPE_NOT_SUPPORTED!\n",
                __FILE__, __LINE__);
        }
    }

    if(VERBOSE >= 2){
        printf("Eigenvalues: [");
        for(int i=0; i < n_eig_vects; ++i){
            printf("%f", result_eigvals[i]);
            if(i+1 < n_eig_vects) printf(", ");
        }
        printf("]\n");
    }

    cluster_forest_cron.stop();
    if(VERBOSE >= 1){
        printf("Creating cluster with forest takes %lf seconds\n",
               cluster_forest_cron.t_total/1000);
    }

    for(int i=0; i < n_trees; i++){
        tinfo_list[i].free();
    }
    
    // int total_leaves = tinfo.total_leaves;
    // int max_child = tinfo.max_child;
    // thrust::device_vector<int> device_nodes_buckets = tinfo.device_nodes_buckets;
    // thrust::device_vector<int> device_bucket_sizes = tinfo.device_bucket_sizes;

    // *nodes_buckets = (int*)malloc(sizeof(int)*total_leaves*max_child);
    // *bucket_sizes  = (int*)malloc(sizeof(int)*total_leaves);
    // thrust::copy(device_nodes_buckets.begin(),
    //              device_nodes_buckets.begin() + total_leaves*max_child,
    //              *nodes_buckets);
    // thrust::copy(device_bucket_sizes.begin(),
    //              device_bucket_sizes.begin()  + total_leaves,
    //              *bucket_sizes);
    // cudaDeviceSynchronize();
    // tinfo.free();

    thrust::copy(result_clustering,
                 result_clustering + N, // Ignore cluster nodes
                 result);
    cudaDeviceSynchronize();
    delete [] edgevals;
    delete [] result_clustering;
    delete [] result_eigvals;
    delete [] result_eigvecs;

    return 0;
}
*/

void RSFK::update_knn_indice_with_buckets(
    thrust::device_vector<RSFK_typepoints> &device_points,
    thrust::device_vector<int> &device_knn_indices,
    thrust::device_vector<RSFK_typepoints> &device_knn_sqr_distances,
    int K, int N, int D, int VERBOSE, TreeInfo tinfo,
    ForestLog& forest_log,
    std::string run_name="out.png")
{
    int total_leaves = tinfo.total_leaves;
    int max_child = tinfo.max_child;
    thrust::device_vector<int> device_nodes_buckets = tinfo.device_nodes_buckets;
    thrust::device_vector<int> device_bucket_sizes = tinfo.device_bucket_sizes;

    // TODO: Automatically specify another id of GPU device rather than 0
    // Automatically get the ideal number of threads per block and total of blocks
    // used in kernels
    int devUsed = 0;
    cudaSetDevice(devUsed);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, devUsed);
    const int NT = deviceProp.maxThreadsPerBlock;
    const int NB = deviceProp.multiProcessorCount*(deviceProp.maxThreadsPerMultiProcessor/deviceProp.maxThreadsPerBlock);


    // TODO: implement correct cudaFuncSetCacheConfig 
    // cudaFuncSetCacheConfig(compute_knn_from_buckets, cudaFuncCachePreferL1);
    // cudaFuncSetCacheConfig(compute_knn_from_buckets_coalesced, cudaFuncCachePreferL1);
    // cudaFuncSetCacheConfig(compute_knn_from_buckets_perwarp_coalesced, cudaFuncCachePreferL1);
    // cudaFuncSetCacheConfig(compute_knn_from_buckets_perblock_coalesced_symmetric_dividek, cudaFuncCachePreferL1);
    // cudaFuncSetCacheConfig(compute_knn_from_buckets_pertile, cudaFuncCachePreferL1);

    Cron cron_knn;
    cron_knn.start();

    // TODO: Check if it is viable to use shared memory 
    
    // compute_knn_from_buckets_perblock_coalesced_symmetric_dividek<<<total_leaves,NT>>>(
    // compute_knn_from_buckets_pertile_coalesced_symmetric<<<total_leaves,32>>>(
    // compute_knn_from_buckets_predist_nolock<<<total_leaves,NT>>>(
    // compute_knn_from_buckets_pertile<<<total_leaves,NT>>>(
    compute_knn_from_buckets_pertile<<<total_leaves,512>>>(
    // compute_knn_from_buckets_pertile<<<total_leaves,32>>>(
    // compute_knn_from_buckets_pertile<<<1,32>>>(
        thrust::raw_pointer_cast(device_points.data()),
        thrust::raw_pointer_cast(device_nodes_buckets.data()),
        thrust::raw_pointer_cast(device_bucket_sizes.data()),
        thrust::raw_pointer_cast(device_knn_indices.data()),
        thrust::raw_pointer_cast(device_knn_sqr_distances.data()),
        N, D, max_child, K, MAX_TREE_CHILD, total_leaves);
    cudaDeviceSynchronize();
    // printf("%s\n", cudaGetErrorString(cudaPeekAtLastError()));
    // printf("%s\n", cudaGetErrorString(cudaThreadSynchronize()));
    CudaTest((char *)"compute_knn_from_buckets Kernel failed!");
    cron_knn.stop();    

    tinfo.free();

    // Report total time of each step
    if(VERBOSE >= 1){
        printf("KNN computation Kernel takes %lf seconds\n", cron_knn.t_total/1000);
    }

    forest_log.update_cron_knn_list((float)cron_knn.t_total/1000);
}


void RSFK::knn_gpu_rsfk_forest(int n_trees,
                               int K, int N, int D, int VERBOSE,
                               std::string run_name="tree")
{
    Cron forest_total_cron;
    forest_total_cron.start();
    thrust::device_vector<RSFK_typepoints> device_points(points, points+N*D);
    
    // thrust::device_vector<RSFK_typepoints> device_points(4096);
    // cudaDeviceSynchronize();

    /*
    thrust::host_vector<int> H(points, points+N*D);

    thrust::device_vector<RSFK_typepoints> device_points(N*D);
    std::cout << "1 ############# " << N << " " << D <<  std::endl;

    thrust::copy(H.begin(), H.end(), device_points.begin());

    cudaDeviceSynchronize();
    */
    
    // thrust::device_vector<RSFK_typepoints> device_points(N*(D+20), 0.0f);
    // for(int i=0; i < N; ++i){
    //     thrust::copy(points+i*D, points+(i+1)*D, device_points.begin()+i*(D+20));
    // }
        
    // std::cout << "2 #############" << std::endl;
    // cudaDeviceSynchronize();

    thrust::device_vector<int> device_knn_indices(knn_indices, knn_indices+N*K);
    
    // std::cout << "3 #############" << std::endl;
    
    thrust::device_vector<RSFK_typepoints> device_knn_sqr_distances(knn_sqr_distances, knn_sqr_distances+N*K);
    
    // std::cout << "4 #############" << std::endl;

    TreeInfo tinfo;
    ForestLog forest_log = ForestLog(n_trees);
    for(int i=0; i < n_trees; ++i){
        tinfo = create_bucket_from_sample_tree(device_points,
                                               N, D, VERBOSE-1,
                                               forest_log,
                                               run_name+"_"+std::to_string(i)+".png",
                                               true, nullptr);

        update_knn_indice_with_buckets(device_points,
                                       device_knn_indices,
                                       device_knn_sqr_distances,
                                       K, N, D, VERBOSE-1, tinfo,
                                       forest_log,
                                       run_name+"_"+std::to_string(i)+".png");

        RANDOM_SEED++;
    }

    forest_total_cron.stop();
    if(VERBOSE >= 1){
        printf("Creating RSFK forest takes %lf seconds\n", forest_total_cron.t_total/1000);
    }

    
    Cron cron_nearest_neighbors_exploring;
    cron_nearest_neighbors_exploring.start();
    
    // TODO: Check if -1 default index will affect nearest neighbor exploration 
    if(nn_exploring_factor > 0){
        thrust::device_vector<int> device_old_knn_indices(knn_indices, knn_indices+N*K);
        // TODO: Automatically specify another id of GPU device rather than 0
        // Automatically get the ideal number of threads per block and total of blocks
        // used in kernels
        int devUsed = 0;
        cudaSetDevice(devUsed);
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, devUsed);
        const int NT = deviceProp.maxThreadsPerBlock;
        const int NB = deviceProp.multiProcessorCount*(deviceProp.maxThreadsPerMultiProcessor/deviceProp.maxThreadsPerBlock);
    
        if(VERBOSE >= 2){
            for(int i=0; i < 80*4; ++i) std::cout<<" ";
            std::cout << std::endl;
            std::cout << "\e[ANearest Neighbor Exploring: " << "0/" << nn_exploring_factor << std::endl;
        }
        for(int i=0; i < nn_exploring_factor; ++i){
            thrust::copy(device_knn_indices.begin(), device_knn_indices.begin() + K*N, device_old_knn_indices.begin());
            cudaDeviceSynchronize();
            nearest_neighbors_exploring<<<NB,NT>>>(thrust::raw_pointer_cast(device_points.data()),
                                                   thrust::raw_pointer_cast(device_old_knn_indices.data()),
                                                   thrust::raw_pointer_cast(device_knn_indices.data()),
                                                   thrust::raw_pointer_cast(device_knn_sqr_distances.data()),
                                                   N, D, K);
            cudaDeviceSynchronize();
            // printf("%s\n", cudaGetErrorString(cudaPeekAtLastError()));
            // printf("%s\n", cudaGetErrorString(cudaThreadSynchronize()));
            cudaGetErrorString(cudaPeekAtLastError());
            cudaGetErrorString(cudaThreadSynchronize());
            CudaTest((char *)"nearest_neighbors_exploring Kernel failed!");
            if(VERBOSE >= 2) std::cout << "\e[ANearest Neighbor Exploring: " << (i+1) << "/" << nn_exploring_factor << std::endl;
        }
        device_old_knn_indices.clear();
        device_old_knn_indices.shrink_to_fit();
    }
    cron_nearest_neighbors_exploring.stop();
    if(VERBOSE >= 1){
        printf("Nearest Neighbors Exploring computation Kernel takes %lf seconds\n", cron_nearest_neighbors_exploring.t_total/1000);
    }

    forest_log.rsfk_total_cron = forest_total_cron.t_total/1000;
    forest_log.nn_exploration_cron = cron_nearest_neighbors_exploring.t_total/1000;

    thrust::copy(device_knn_indices.begin(), device_knn_indices.begin() + N*K, knn_indices);
    thrust::copy(device_knn_sqr_distances.begin(), device_knn_sqr_distances.begin() + N*K, knn_sqr_distances);
    cudaDeviceSynchronize();

    float tmp_log;
    for(int i=0; i < 16; ++i){
        for(int j=0; j < n_trees; ++j){
            switch(i){
                case(0): tmp_log = forest_log.max_depth_list[j]; break;
                case(1): tmp_log = forest_log.max_child_count_list[j]; break;
                case(2): tmp_log = forest_log.min_child_count_list[j]; break;
                case(3): tmp_log = forest_log.total_leaves_list[j]; break;
                case(4): tmp_log = forest_log.init_tree_cron_list[j]; break;
                case(5): tmp_log = forest_log.total_tree_build_cron_list[j]; break;
                case(6): tmp_log = forest_log.check_active_points_cron_list[j]; break;
                case(7): tmp_log = forest_log.check_points_side_cron_list[j]; break;
                case(8): tmp_log = forest_log.tree_count_cron_list[j]; break;
                case(9): tmp_log = forest_log.dynamic_memory_allocation_cron_list[j]; break;
                case(10): tmp_log = forest_log.organize_sample_candidate_cron_list[j]; break;
                case(11): tmp_log = forest_log.create_nodes_cron_list[j]; break;
                case(12): tmp_log = forest_log.update_parents_cron_list[j]; break;
                case(13): tmp_log = forest_log.cron_classify_points_list[j]; break;
                case(14): tmp_log = forest_log.end_tree_cron_list[j]; break;
                case(15): tmp_log = forest_log.cron_knn_list[j]; break;
            }
            log_forest_output[i*n_trees+j] = tmp_log;
        }
    }
    log_forest_output[16*n_trees] = forest_log.rsfk_total_cron;
    log_forest_output[16*n_trees + 1] = forest_log.nn_exploration_cron;
    
    device_points.clear();
    device_points.shrink_to_fit();
    device_knn_indices.clear();
    device_knn_indices.shrink_to_fit();
    device_knn_sqr_distances.clear();
    device_knn_sqr_distances.shrink_to_fit();
    
    forest_log.free();
}

TreeInfo RSFK::cluster_by_sample_tree(int N, int D, int VERBOSE,
                                      int** nodes_buckets,
                                      int** bucket_sizes,
                                      std::string run_name="tree_cluster")
{
    Cron cluster_forest_cron;
    cluster_forest_cron.start();
    thrust::device_vector<RSFK_typepoints> device_points(points, points+N*D);
    
    TreeInfo tinfo;
    ForestLog forest_log = ForestLog(1);
    tinfo = create_bucket_from_sample_tree(device_points,
                                           N, D, VERBOSE-1,
                                           forest_log,
                                           run_name+".png",
                                           true, nullptr);

    cluster_forest_cron.stop();
    if(VERBOSE >= 1){
        printf("Creating cluster with one tree takes %lf seconds\n",
               cluster_forest_cron.t_total/1000);
    }

    int total_leaves = tinfo.total_leaves;
    int max_child = tinfo.max_child;
    thrust::device_vector<int> device_nodes_buckets = tinfo.device_nodes_buckets;
    thrust::device_vector<int> device_bucket_sizes = tinfo.device_bucket_sizes;

    *nodes_buckets = (int*)malloc(sizeof(int)*total_leaves*max_child);
    *bucket_sizes  = (int*)malloc(sizeof(int)*total_leaves);
    thrust::copy(device_nodes_buckets.begin(),
                 device_nodes_buckets.begin() + total_leaves*max_child,
                 *nodes_buckets);
    thrust::copy(device_bucket_sizes.begin(),
                 device_bucket_sizes.begin()  + total_leaves,
                 *bucket_sizes);
    cudaDeviceSynchronize();
    tinfo.free();

    device_points.clear();
    device_points.shrink_to_fit();

    return tinfo;
}

int main(int argc,char* argv[])
{    
    if(argc < 6){
        std::cout << "Error. run with ./binary <Total of points> <Dimensions> <Max Depth> <Verbose> <Dataset ID>" << std::endl;
        return 1;
    }

    int RANDOM_SEED = 0;

    srand(RANDOM_SEED);

    int N = atoi(argv[1]);
    int D = atoi(argv[2]);
    int MAX_DEPTH = atoi(argv[3]);
    int VERBOSE = atoi(argv[4]);
    int DS = atoi(argv[5]);

    RSFK_typepoints* points = (RSFK_typepoints*) malloc(sizeof(RSFK_typepoints)*N*D);
    int K = 32;

    int* knn_indices = (int*) malloc(sizeof(int)*N*K);
    std::fill_n(knn_indices, N*K, -1);

    RSFK_typepoints* knn_sqr_distances = (RSFK_typepoints*) malloc(sizeof(RSFK_typepoints)*N*K);
    std::fill_n(knn_sqr_distances, N*K, FLT_MAX);


    std::vector<int> labels(N*D);

    int l;
    std::string type_init;
    if(DS == 0){
        type_init = "SPLITED_LINE";
    }
    if(DS == 1){
        type_init = "UNIFORM_SQUARE";
    }

    for(int i=0; i < N; ++i){
        if(type_init == "UNIFORM_SQUARE"){
            l = rand() % N;
        }
        if(type_init == "SPLITED_LINE"){
            l = i;
        }

        for(int j=0; j < D; ++j){
            if(type_init == "UNIFORM_SQUARE"){
                points[get_point_idx(i,j,N,D)] = rand() % N;
            }
            if(type_init == "SPLITED_LINE"){
                points[get_point_idx(i,j,N,D)] = l + (l>N/2)*N/2;
            }
        }
        labels[i] = (l>N/2);
    }

    int nn_exploring_factor = 0;
    float* forest_log_output = (float*)malloc(sizeof(float)*5*16+2);

    RSFK rsfk_knn(points, nullptr, knn_indices, knn_sqr_distances, K+1, 2*(K+1), MAX_DEPTH,
                  RANDOM_SEED, nn_exploring_factor, forest_log_output);
    rsfk_knn.knn_gpu_rsfk_forest(5, K, N, D, VERBOSE, "tree");

    return 0;
}
#endif