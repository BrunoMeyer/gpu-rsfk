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

#ifndef __RSFK__H
#define __RSFK__H


// #include "nvgraph.h"

// Thrust includes
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/random.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/functional.h>
#include <thrust/fill.h>
#include <thrust/gather.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>

// CUDA includes
#include <curand.h>
#include <curand_kernel.h>




#include <iostream> 
#include <stdio.h>
#include <cstdlib>
#include <cmath>
#include <bits/stdc++.h> 

#include "common.h"

#include "kernels/build_tree_bucket_points.h"
#include "kernels/build_tree_check_points_side.h"
#include "kernels/build_tree_count_new_nodes.h"
#include "kernels/build_tree_create_nodes.h"
#include "kernels/build_tree_init.h"
#include "kernels/build_tree_update_parents.h"
#include "kernels/build_tree_utils.h"
#include "kernels/compute_knn_from_buckets.h"
#include "kernels/nearest_neighbors_exploring.h"
#include "kernels/build_tree_update_parents_ann.h"


class RSFKIndexTree{
public:
    thrust::device_vector<RSFK_typepoints>** device_tree;
    thrust::device_vector<int>** device_tree_parents;
    thrust::device_vector<int>** device_tree_children;
    thrust::device_vector<bool>** device_is_leaf;
    thrust::device_vector<int>** device_child_count;
    thrust::device_vector<int>** device_accumulated_child_count;
    thrust::device_vector<int>** device_count_points_on_leaves;

    thrust::device_vector<int>* device_points_parent;
    thrust::device_vector<int>* device_points_depth;
    thrust::device_vector<int>* device_is_right_child;
    thrust::device_vector<int>* device_sample_candidate_points;

    thrust::device_vector<int>* device_points_id_on_sample;
    thrust::device_vector<int>* device_active_points;

    thrust::device_vector<int>* device_depth_level_count;
    thrust::device_vector<int>* device_accumulated_nodes_count;
    thrust::device_vector<int>* device_tree_count;
    thrust::device_vector<int>* device_count_new_nodes;
    thrust::device_vector<int>* device_actual_depth;
    thrust::device_vector<int>* device_active_points_count;

    thrust::device_vector<int>* device_leaf_idx_to_node_idx;
    thrust::device_vector<int>* device_node_idx_to_leaf_idx;
    thrust::device_vector<int>* device_nodes_buckets;
    thrust::device_vector<int>* device_bucket_sizes;
    
    thrust::device_vector<int>* device_max_leaf_size;
    thrust::device_vector<int>* device_min_leaf_size;
    thrust::device_vector<int>* device_total_leaves;

    int total_leaves;
    int max_child;
    int reached_max_depth;
    int MAX_TREE_CHILD;
    int total_buckets;

    RSFKIndexTree(){
        device_tree = nullptr;
        device_tree_parents = nullptr;
        device_tree_children = nullptr;
        device_is_leaf = nullptr;
        device_child_count = nullptr;
        device_accumulated_child_count = nullptr;
        device_count_points_on_leaves = nullptr;

        device_points_parent = nullptr;
        device_points_depth = nullptr;
        device_is_right_child = nullptr;
        device_sample_candidate_points = nullptr;

        device_points_id_on_sample = nullptr;
        device_active_points = nullptr;

        device_depth_level_count = nullptr;
        device_accumulated_nodes_count = nullptr;
        device_tree_count = nullptr;
        device_count_new_nodes = nullptr;
        device_actual_depth = nullptr;
        device_active_points_count = nullptr;

        device_leaf_idx_to_node_idx = nullptr;
        device_node_idx_to_leaf_idx = nullptr;
        device_nodes_buckets = nullptr;
        device_bucket_sizes = nullptr;

        device_max_leaf_size = nullptr;
        device_min_leaf_size = nullptr;
        device_total_leaves = nullptr;

        total_leaves = -1;
        max_child = -1;
        reached_max_depth = -1;

        MAX_TREE_CHILD = -1;
        total_buckets = -1;
        
    }

    void free(){
        device_leaf_idx_to_node_idx->clear();
        device_leaf_idx_to_node_idx->shrink_to_fit();
        device_node_idx_to_leaf_idx->clear();
        device_node_idx_to_leaf_idx->shrink_to_fit();
        device_nodes_buckets->clear();
        device_nodes_buckets->shrink_to_fit();
        device_bucket_sizes->clear();
        device_bucket_sizes->shrink_to_fit();
        device_points_parent->clear();
        device_points_parent->shrink_to_fit();
        device_points_depth->clear();
        device_points_depth->shrink_to_fit();
        device_is_right_child->clear();
        device_is_right_child->shrink_to_fit();
        device_sample_candidate_points->clear();
        device_sample_candidate_points->shrink_to_fit();
        device_points_id_on_sample->clear();
        device_points_id_on_sample->shrink_to_fit();

        device_depth_level_count->clear();
        device_depth_level_count->shrink_to_fit();
        device_accumulated_nodes_count->clear();
        device_accumulated_nodes_count->shrink_to_fit();
        device_tree_count->clear();
        device_tree_count->shrink_to_fit();
        device_count_new_nodes->clear();
        device_count_new_nodes->shrink_to_fit();
        device_actual_depth->clear();
        device_actual_depth->shrink_to_fit();
        
        device_max_leaf_size->clear();
        device_max_leaf_size->shrink_to_fit();
        device_min_leaf_size->clear();
        device_min_leaf_size->shrink_to_fit();
        device_total_leaves->clear();
        device_total_leaves->shrink_to_fit();
        device_active_points->clear();
        device_active_points->shrink_to_fit();

        for(int depth=0; depth < reached_max_depth; ++depth){
            // Random Projection Forest
            // device_random_directions[depth]->clear();
            // device_random_directions[depth]->shrink_to_fit();
            // device_min_random_proj_values[depth]->clear();
            // device_min_random_proj_values[depth]->shrink_to_fit();
            // device_max_random_proj_values[depth]->clear();
            // device_max_random_proj_values[depth]->shrink_to_fit();
            device_tree[depth]->clear();
            device_tree[depth]->shrink_to_fit();
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

    // void update_knn_indice_with_buckets(thrust::device_vector<RSFK_typepoints> &device_points,
    //                                     thrust::device_vector<int> &device_knn_indices,
    //                                     thrust::device_vector<RSFK_typepoints> &device_knn_sqr_distances,
    //                                     int K, int N, int D, int VERBOSE, TreeInfo tinfo,
    //                                     ForestLog& forest_log,
    //                                     std::string run_name);
};

class TreeInfo{
public:
    int total_leaves;
    int max_child;

    thrust::device_vector<int> device_nodes_buckets;
    thrust::device_vector<int> device_bucket_sizes;
    
    TreeInfo(){}

    TreeInfo(int total_leaves,
             int max_child,
             thrust::device_vector<int> &_device_nodes_buckets,
             thrust::device_vector<int> &_device_bucket_sizes):
             total_leaves(total_leaves),
             max_child(max_child)
    {
        device_nodes_buckets = _device_nodes_buckets;
        device_bucket_sizes = _device_bucket_sizes;
    }

    void free(){
        device_nodes_buckets.clear();
        device_nodes_buckets.shrink_to_fit();
        device_bucket_sizes.clear();
        device_bucket_sizes.shrink_to_fit();
    }
};


class ForestLog{
public:
    int n_trees;
    int count_tree;


    int* max_depth_list;
    int* max_child_count_list;
    int* min_child_count_list;
    int* total_leaves_list;
    float* init_tree_cron_list;
    float* total_tree_build_cron_list;
    float* check_active_points_cron_list;
    float* check_points_side_cron_list;
    float* tree_count_cron_list;
    float* dynamic_memory_allocation_cron_list;
    float* organize_sample_candidate_cron_list;
    float* create_nodes_cron_list;
    float* update_parents_cron_list;
    float* cron_classify_points_list;
    float* end_tree_cron_list;
    float* cron_knn_list;

    float nn_exploration_cron;
    float rsfk_total_cron;

    ForestLog(){}

    ForestLog(int n_trees):
              n_trees(n_trees)
    {
        count_tree = 0;
        
        max_depth_list = (int*)malloc(sizeof(int)*n_trees);
        max_child_count_list = (int*)malloc(sizeof(int)*n_trees);
        min_child_count_list = (int*)malloc(sizeof(int)*n_trees);
        total_leaves_list = (int*)malloc(sizeof(int)*n_trees);
        init_tree_cron_list = (float*)malloc(sizeof(float)*n_trees);
        total_tree_build_cron_list = (float*)malloc(sizeof(float)*n_trees);
        check_active_points_cron_list = (float*)malloc(sizeof(float)*n_trees);
        check_points_side_cron_list = (float*)malloc(sizeof(float)*n_trees);
        tree_count_cron_list = (float*)malloc(sizeof(float)*n_trees);
        dynamic_memory_allocation_cron_list = (float*)malloc(sizeof(float)*n_trees);
        organize_sample_candidate_cron_list = (float*)malloc(sizeof(float)*n_trees);
        create_nodes_cron_list = (float*)malloc(sizeof(float)*n_trees);
        update_parents_cron_list = (float*)malloc(sizeof(float)*n_trees);
        cron_classify_points_list = (float*)malloc(sizeof(float)*n_trees);
        end_tree_cron_list = (float*)malloc(sizeof(float)*n_trees);
        cron_knn_list = (float*)malloc(sizeof(float)*n_trees);
    }

    void free(){
        std::free(max_depth_list);
        std::free(max_child_count_list);
        std::free(min_child_count_list);
        std::free(total_leaves_list);
        std::free(init_tree_cron_list);
        std::free(total_tree_build_cron_list);
        std::free(check_active_points_cron_list);
        std::free(check_points_side_cron_list);
        std::free(tree_count_cron_list);
        std::free(dynamic_memory_allocation_cron_list);
        std::free(organize_sample_candidate_cron_list);
        std::free(create_nodes_cron_list);
        std::free(update_parents_cron_list);
        std::free(cron_classify_points_list);
        std::free(end_tree_cron_list);
        std::free(cron_knn_list);
    }

    void update_cron_knn_list(float time)
    {
        cron_knn_list[count_tree-1] = time;
    }

    void update_cron_knn_list(float time, bool update_counter)
    {
        cron_knn_list[count_tree-1] = time;
        if(update_counter) ++count_tree;
    }

    void update_log(
        int max_depth,
        int max_child_count,
        int min_child_count,
        int total_leaves,
        float init_tree_cron,
        float total_tree_build_cron,
        float check_active_points_cron,
        float check_points_side_cron,
        float tree_count_cron,
        float dynamic_memory_allocation_cron,
        float organize_sample_candidate_cron,
        float create_nodes_cron,
        float update_parents_cron,
        float cron_classify_points,
        float end_tree_cron)
    {
        max_depth_list[count_tree] = max_depth;
        max_child_count_list[count_tree] = max_child_count;
        min_child_count_list[count_tree] = min_child_count;
        total_leaves_list[count_tree] = total_leaves;
        init_tree_cron_list[count_tree] = init_tree_cron;
        total_tree_build_cron_list[count_tree] = total_tree_build_cron;
        check_active_points_cron_list[count_tree] = check_active_points_cron;
        check_points_side_cron_list[count_tree] = check_points_side_cron;
        tree_count_cron_list[count_tree] = tree_count_cron;
        dynamic_memory_allocation_cron_list[count_tree] = dynamic_memory_allocation_cron;
        organize_sample_candidate_cron_list[count_tree] = organize_sample_candidate_cron;
        create_nodes_cron_list[count_tree] = create_nodes_cron;
        update_parents_cron_list[count_tree] = update_parents_cron;
        cron_classify_points_list[count_tree] = cron_classify_points;
        end_tree_cron_list[count_tree] = end_tree_cron;

        count_tree++;
    }


};

// Class used to construct Random Projection forest and execute KNN
class RSFK
{
public:
    
    // Data points. It will be stored as POINTS x DIMENSION or
    // DIMENSION x POINTS considering the defined POINTS_STRUCTURE
    RSFK_typepoints* points; 
    RSFK_typepoints* query_points; 
    
    // Indices of the estimated k-nearest neighbors for each point
    // and squared distances between
    // It can be previously initialized before the execution of RSFK
    // with valid indices and distances, otherwise it must assume that indices
    // have -1 value and distances FLT_MAX (or DBL_MAX)
    // The indices ARE NOT sorted by the relative distances
    int* knn_indices;
    RSFK_typepoints* knn_sqr_distances;
    
    // Maximum and Minimum number of points that will be present in each leaf node (bucket)
    // This affect the local KNN step after the construction of each tree
    int MIN_TREE_CHILD;
    int MAX_TREE_CHILD;

    // The limit of the tree depth. The algorithm may break if a tree reach
    // a higher value than this parameter. When the tree is 'ready',
    // there is a early stop and this limit will not be reached
    // Known Bug: It may be necessary to add a 'random motion' before the
    // execution of the algorithm to prevent that two points be positioned
    // in the the same location, leading to the impossibility of the
    // tree construction and the reach of MAX_DEPTH 
    int MAX_DEPTH;

    // SEED used to the pseudorandom number generator
    int RANDOM_SEED;

    // Nearest neighbor exploring is a pos-processing technique that will improve
    // the accuracy of the approximated KNN. The neighbors of the neighbors
    // will be treated as neighbor candidates for each point, which is a
    // O(N*D*(K^2)) time cost step. This can be executed several times using the
    // updated indices as the new input, and this ammount is represented by
    // nn_exploring_factor. When nn_exploring_factor=0 the Nearest Neighbor Exploring
    // will not be executed
    int nn_exploring_factor;

    float* log_forest_output;
    
    RSFK(RSFK_typepoints* points,
         RSFK_typepoints* query_points,
         int* knn_indices,
         RSFK_typepoints* knn_sqr_distances,
         int MIN_TREE_CHILD,
         int MAX_TREE_CHILD,
         int MAX_DEPTH,
         int RANDOM_SEED,
         int nn_exploring_factor,
         float* log_forest_output):
         points(points),
         query_points(query_points),
         knn_indices(knn_indices),
         knn_sqr_distances(knn_sqr_distances),
         MIN_TREE_CHILD(MIN_TREE_CHILD),
         MAX_TREE_CHILD(MAX_TREE_CHILD),
         MAX_DEPTH(MAX_DEPTH),
         RANDOM_SEED(RANDOM_SEED),
         nn_exploring_factor(nn_exploring_factor),
         log_forest_output(log_forest_output){}
    

    // Create a random projection tree and update the indices by considering
    // the points in the same leaf node as candidates to neighbor
    // This ensure that each point will have K valid neighbors indices, which
    // can be very inaccurate.
    // The device_knn_indices parameter can be previously initialized with
    // valid indices or -1 values. If it has valid indices, also will be necessary
    // to add the precomputed squared distances (device_knn_sqr_distances) 
    TreeInfo create_bucket_from_sample_tree(thrust::device_vector<RSFK_typepoints> &device_points,
                                            int N, int D, int VERBOSE,
                                            ForestLog& forest_log,
                                            std::string run_name,
                                            bool free_index,
                                            RSFKIndexTree* rsfkindextree);

    void update_knn_indice_with_buckets(thrust::device_vector<RSFK_typepoints> &device_points,
                                        thrust::device_vector<int> &device_knn_indices,
                                        thrust::device_vector<RSFK_typepoints> &device_knn_sqr_distances,
                                        int K, int N, int D, int VERBOSE, TreeInfo tinfo,
                                        ForestLog& forest_log,
                                        std::string run_name);
    
    // Run n_tree times the add_random_projection_tree procedure and the nearest
    // neighbors exploring if necessary
    void knn_gpu_rsfk_forest(int n_trees,
                             int K, int N, int D, int VERBOSE,
                             std::string run_name);

    // Run n_tree times the add_random_projection_tree procedure and the nearest
    // neighbors exploring if necessary
    TreeInfo cluster_by_sample_tree(int N, int D, int VERBOSE,
                                    int** nodes_buckets,
                                    int** bucket_sizes,
                                    std::string run_name);

    int create_cluster_with_hbgf(int* result, int n_trees,
                                 int N, int D, int VERBOSE,
                                 int K, int n_eig_vects,
                                 std::string run_name);

    int spectral_clustering_with_knngraph(int* result, int num_neighbors,
                                          int N, int D, int VERBOSE,
                                          int K, int n_eig_vects,
                                          bool free_knn_indices,
                                          std::string run_name);


    void knn_gpu_rsfk_forest_ann(
        int n_trees,
        int K, int N, int NQ, int D, int VERBOSE,
        std::string run_name);
    
    void knn_gpu_rsfk_forest_ann_tree(
        int n_trees,
        thrust::device_vector<RSFK_typepoints> &device_points,
        thrust::device_vector<RSFK_typepoints> &device_query_points,
        thrust::device_vector<RSFK_typepoints> &device_knn_sqr_distances,
        thrust::device_vector<int> &device_knn_indices,
        int K, int N, int NQ, int D, int VERBOSE,
        std::string run_name,
        RSFKIndexTree* rsfkindextree);
};


#endif