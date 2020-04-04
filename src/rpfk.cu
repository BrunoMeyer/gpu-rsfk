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



#include "third_party/matplotlibcpp.h"
namespace plt = matplotlibcpp;

#include <iostream> 
#include <stdio.h>
using namespace std;
#include <cstdlib>
#include <cmath>
#include <bits/stdc++.h> 



#define typepoints float

// Lines per column
#define N_D 0

// Columns 
#define D_N 1

#define POINTS_STRUCTURE N_D
#define TREE_STRUCTURE N_D

#if   POINTS_STRUCTURE == D_N
    #define get_point_idx(point,dimension,N,D) (dimension*N+point)
#elif POINTS_STRUCTURE == N_D
    #define get_point_idx(point,dimension,N,D) (point*D+dimension)
#endif

#if   TREE_STRUCTURE == D_N
    #define get_tree_idx(nidx,dimension,N,D) (dimension*N+nidx)
#elif TREE_STRUCTURE == N_D
    #define get_tree_idx(nidx,dimension,N,D) (nidx*(D+1)+dimension)
#endif

#define RELEASE 0
#define DEBUG 1
#define COMPILE_TYPE RELEASE

#include "kernels/build_tree_init.cu"
#include "kernels/build_tree_check_points_side.cu"
#include "kernels/build_tree_count_new_nodes.cu"
#include "kernels/build_tree_create_nodes.cu"
#include "kernels/build_tree_update_parents.cu"
#include "kernels/build_tree_utils.cu"
#include "kernels/build_tree_bucket_points.cu"
#include "kernels/compute_knn_from_buckets.cu"
#include "kernels/nearest_neighbors_exploring.cu"



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

// Class used to construct Random Projection forest and execute KNN
class RPFK
{
public:
    
    // Data points. It will be stored as POINTS x DIMENSION or
    // DIMENSION x POINTS considering the defined POINTS_STRUCTURE
    typepoints* points; 
    
    // Indices of the estimated k-nearest neighbors for each point
    // and squared distances between
    // It can be previously initialized before the execution of RPFK
    // with valid indices and distances, otherwise it must assume that indices
    // have -1 value and distances FLT_MAX (or DBL_MAX)
    // The indices ARE NOT sorted by the relative distances
    int* knn_indices;
    typepoints* knn_sqr_distances;
    
    // Maximum number of points that will be present in each leaf node (bucket)
    // This affect the local KNN step after the construction of each tree
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
    
    RPFK(typepoints* points,
         int* knn_indices,
         typepoints* knn_sqr_distances,
         int MAX_TREE_CHILD,
         int MAX_DEPTH,
         int RANDOM_SEED,
         int nn_exploring_factor):
         points(points),
         knn_indices(knn_indices),
         knn_sqr_distances(knn_sqr_distances),
         MAX_TREE_CHILD(MAX_TREE_CHILD),
         MAX_DEPTH(MAX_DEPTH),
         RANDOM_SEED(RANDOM_SEED),
         nn_exploring_factor(nn_exploring_factor){}
    

    // Create a random projection tree and update the indices by considering
    // the points in the same leaf node as candidates to neighbor
    // This ensure that each point will have K valid neighbors indices, which
    // can be very inaccurate.
    // The device_knn_indices parameter can be previously initialized with
    // valid indices or -1 values. If it has valid indices, also will be necessary
    // to add the precomputed squared distances (device_knn_sqr_distances) 
    void add_random_projection_tree(thrust::device_vector<typepoints> &device_points,
                                    thrust::device_vector<int> &device_old_knn_indices,
                                    thrust::device_vector<int> &device_knn_indices,
                                    thrust::device_vector<typepoints> &device_knn_sqr_distances,
                                    int K, int N, int D, int VERBOSE,
                                    string run_name);
    
    
    // Run n_tree times the add_random_projection_tree procedure and the nearest
    // neighbors exploring if necessary
    void knn_gpu_rpfk_forest(int n_trees,
                             int K, int N, int D, int VERBOSE,
                             string run_name);
};

void RPFK::add_random_projection_tree(thrust::device_vector<typepoints> &device_points,
                        thrust::device_vector<int> &device_old_knn_indices,
                        thrust::device_vector<int> &device_knn_indices,
                        thrust::device_vector<typepoints> &device_knn_sqr_distances,
                        int K, int N, int D, int VERBOSE,
                        string run_name="out.png")
{
    Cron init_tree_cron, end_tree_cron, total_tree_build_cron, check_active_points_cron,
         update_parents_cron, create_nodes_cron, check_points_side_cron, tree_count_cron,
         organize_sample_candidate_cron, dynamic_memory_allocation_cron;
    
    init_tree_cron.start();
    
    
         // They are pointers to each vector that represent a level of the tree
    // These vectors are allocated dynamically during the tree construction
    
    // Tree nodes hyperplanes values (equations of size D+1)
    thrust::device_vector<typepoints>** device_tree = (thrust::device_vector<typepoints>**) malloc(sizeof(thrust::device_vector<typepoints>*)*MAX_DEPTH);
    
    
    // thrust::device_vector<typepoints>** device_random_directions = (thrust::device_vector<typepoints>**) malloc(sizeof(thrust::device_vector<typepoints>*)*MAX_DEPTH);
    
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
    device_tree[0] = new thrust::device_vector<typepoints>((D+1)*MAX_NODES);
    // device_random_directions[0] = new thrust::device_vector<typepoints>(2*D*MAX_NODES);
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


    // thrust::device_vector<typepoints> device_min_random_proj_values(N, FLT_MAX);
    // thrust::device_vector<typepoints> device_max_random_proj_values(N, FLT_MIN);

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
    #if COMPILE_TYPE == DEBUG
        thrust::device_vector<int> device_count_undo_leaf(1, 0);
    #endif

    

    // Automatically get the ideal number of threads per block and total of blocks
    // used in kernels
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
        // thrust::fill(thrust::device, device_sample_candidate_points.begin(), device_sample_candidate_points.end(), -1);
        cudaDeviceSynchronize();
        
        check_active_points_cron.start();
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
        build_tree_check_points_side_coalesced<<<NB,NT>>>(thrust::raw_pointer_cast(device_tree[depth-1]->data()),
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
        thrust::fill(thrust::device, device_count_new_nodes.begin(), device_count_new_nodes.end(), 0);
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
                                              N, D, K, MAX_TREE_CHILD);
        cudaDeviceSynchronize();
        CudaTest((char *)"build_tree_count_new_nodes Kernel failed!");
        thrust::copy(device_count_new_nodes.begin(), device_count_new_nodes.begin()+1, &count_new_nodes);
        cudaDeviceSynchronize();
        tree_count_cron.stop();
        
        
        if(count_new_nodes > 0){
            dynamic_memory_allocation_cron.start();
            count_total_nodes+=count_new_nodes;

            device_tree[depth] = new thrust::device_vector<typepoints>(((D+1)*count_new_nodes));
            // device_random_directions[depth] = new thrust::device_vector<typepoints>((2*D*count_new_nodes));
            device_tree_parents[depth] = new thrust::device_vector<int>(count_new_nodes,-1);
            device_tree_children[depth] = new thrust::device_vector<int>(2*count_new_nodes,-1);
            device_is_leaf[depth] = new thrust::device_vector<bool>(count_new_nodes, false);
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
                                               N, D, K, MAX_TREE_CHILD, RANDOM_SEED);
            cudaDeviceSynchronize();
            CudaTest((char *)"build_tree_create_nodes Kernel failed!");
            create_nodes_cron.stop();


            update_parents_cron.start();
            build_tree_update_parents<<<NB,NT>>>(thrust::raw_pointer_cast(device_tree[depth-1]->data()),
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
                                                N, D, K, MAX_TREE_CHILD);
            cudaDeviceSynchronize();
            CudaTest((char *)"build_tree_update_parents Kernel failed!");
            build_tree_post_update_parents<<<NB,NT>>>(thrust::raw_pointer_cast(device_tree[depth-1]->data()),
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
                                                    #if COMPILE_TYPE == DEBUG
                                                        thrust::raw_pointer_cast(device_count_undo_leaf.data()),
                                                    #endif
                                                    N, D, K, MAX_TREE_CHILD);
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

        

        if(VERBOSE >= 1){
            std::cout << "\e[ABuilding Tree Depth: " << depth+1 << "/" << MAX_DEPTH;
            std::cout << " | Total nodes: " << count_total_nodes << std::endl;
        }

        if(count_new_nodes == 0){
            if(VERBOSE >= 1) std::cout << "Early stop: 0 new nodes created." << std::endl;
            break;
        }
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
    
    #if COMPILE_TYPE == DEBUG
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


    std::vector<int> nodes_buckets;
    std::vector<int> bucket_sizes;
    if(VERBOSE >= 3){
        nodes_buckets.resize(total_leaves*max_child);
        thrust::copy(device_nodes_buckets.begin(), device_nodes_buckets.begin() + total_leaves*max_child, nodes_buckets.begin());
        bucket_sizes.resize(total_leaves);
        thrust::copy(device_bucket_sizes.begin(), device_bucket_sizes.begin() + total_leaves, bucket_sizes.begin());
        cudaDeviceSynchronize();
    }

    
    
    

    // cudaFuncSetCacheConfig(compute_knn_from_buckets, cudaFuncCachePreferL1);
    cudaFuncSetCacheConfig(compute_knn_from_buckets_coalesced, cudaFuncCachePreferL1);
    cudaFuncSetCacheConfig(compute_knn_from_buckets_perwarp_coalesced, cudaFuncCachePreferL1);

    Cron cron_knn;
    cron_knn.start();

    // TODO: Check if it is viable to use shared memory 
    // compute_knn_from_buckets_small_block<<<mp*nt/32,32, 32*D*sizeof(typepoints)>>>(thrust::raw_pointer_cast(device_points_parent.data()),
    //                                     thrust::raw_pointer_cast(device_points_depth.data()),
    //                                     thrust::raw_pointer_cast(device_accumulated_nodes_count.data()),
    //                                     thrust::raw_pointer_cast(device_points.data()),
    //                                     thrust::raw_pointer_cast(device_node_idx_to_leaf_idx.data()),
    //                                     thrust::raw_pointer_cast(device_nodes_buckets.data()),
    //                                     thrust::raw_pointer_cast(device_bucket_sizes.data()),
    //                                     thrust::raw_pointer_cast(device_knn_indices.data()),
    //                                     thrust::raw_pointer_cast(device_knn_sqr_distances.data()),
    //                                     N, D, max_child, K, MAX_TREE_CHILD, total_leaves);
    // compute_knn_from_buckets<<<NB,NT>>>(thrust::raw_pointer_cast(device_points_parent.data()),
    //                                     thrust::raw_pointer_cast(device_points_depth.data()),
    //                                     thrust::raw_pointer_cast(device_accumulated_nodes_count.data()),
    //                                     thrust::raw_pointer_cast(device_points.data()),
    //                                     thrust::raw_pointer_cast(device_node_idx_to_leaf_idx.data()),
    //                                     thrust::raw_pointer_cast(device_nodes_buckets.data()),
    //                                     thrust::raw_pointer_cast(device_bucket_sizes.data()),
    //                                     thrust::raw_pointer_cast(device_knn_indices.data()),
    //                                     thrust::raw_pointer_cast(device_knn_sqr_distances.data()),
    //                                     N, D, max_child, K, MAX_TREE_CHILD, total_leaves);
    // compute_knn_from_buckets_coalesced<<<NB,NT, sizeof(typepoints)*((1024/32)+K) + sizeof(int)*K>>>(
    //                                               thrust::raw_pointer_cast(device_points_parent.data()),
    //                                               thrust::raw_pointer_cast(device_points_depth.data()),
    //                                               thrust::raw_pointer_cast(device_accumulated_nodes_count.data()),
    //                                               thrust::raw_pointer_cast(device_points.data()),
    //                                               thrust::raw_pointer_cast(device_node_idx_to_leaf_idx.data()),
    //                                               thrust::raw_pointer_cast(device_nodes_buckets.data()),
    //                                               thrust::raw_pointer_cast(device_bucket_sizes.data()),
    //                                               thrust::raw_pointer_cast(device_knn_indices.data()),
    //                                               thrust::raw_pointer_cast(device_knn_sqr_distances.data()),
    //                                               N, D, max_child, K, MAX_TREE_CHILD);
    compute_knn_from_buckets_perwarp_coalesced<<<NB,NT, NT*sizeof(typepoints)>>>(
                                              thrust::raw_pointer_cast(device_points_parent.data()),
                                              thrust::raw_pointer_cast(device_points_depth.data()),
                                              thrust::raw_pointer_cast(device_accumulated_nodes_count.data()),
                                              thrust::raw_pointer_cast(device_points.data()),
                                              thrust::raw_pointer_cast(device_node_idx_to_leaf_idx.data()),
                                              thrust::raw_pointer_cast(device_nodes_buckets.data()),
                                              thrust::raw_pointer_cast(device_bucket_sizes.data()),
                                              thrust::raw_pointer_cast(device_knn_indices.data()),
                                              thrust::raw_pointer_cast(device_knn_sqr_distances.data()),
                                              N, D, max_child, K, MAX_TREE_CHILD, total_leaves);
    // compute_knn_from_buckets_old<<<NB,NT>>>(thrust::raw_pointer_cast(device_points_parent.data()),
    //                                         thrust::raw_pointer_cast(device_points_depth.data()),
    //                                         thrust::raw_pointer_cast(device_accumulated_nodes_count.data()),
    //                                         thrust::raw_pointer_cast(device_points.data()),
    //                                         thrust::raw_pointer_cast(device_node_idx_to_leaf_idx.data()),
    //                                         thrust::raw_pointer_cast(device_nodes_buckets.data()),
    //                                         thrust::raw_pointer_cast(device_bucket_sizes.data()),
    //                                         thrust::raw_pointer_cast(device_knn_indices.data()),
    //                                         thrust::raw_pointer_cast(device_knn_sqr_distances.data()),
    //                                         N, D, max_child, K, MAX_TREE_CHILD);
    cudaDeviceSynchronize();
    CudaTest((char *)"compute_knn_from_buckets Kernel failed!");
    cron_knn.stop();    
    end_tree_cron.start();
    
    

    // cudaDeviceSynchronize();

    if(VERBOSE >= 2){
        thrust::copy(device_knn_indices.begin(), device_knn_indices.begin() + N*K, knn_indices);
        thrust::copy(device_knn_sqr_distances.begin(), device_knn_sqr_distances.begin() + N*K, knn_sqr_distances);
        std::cout << "Points neighbors" << std::endl;
        for(int i=0; i < 1*16; i+=1){
            std::cout << i << ": ";
            for(int j=0; j < K; ++j) std::cout << knn_indices[i*K+j] << " ";
            std::cout << std::endl;
        }
        for(int i=N-4; i < N; ++i){
            std::cout << i << ": ";
            for(int j=0; j < K; ++j) std::cout << knn_indices[i*K+j] << " ";
            std::cout << std::endl;
        }
        // std::cout << "Bucket HEAP" << std::endl;
        // for(int i=MAX_NODES-1; i >= MAX_NODES-8; --i){
        //     // std::cout << i << "(" << bucket_sizes[i] << ") : ";
        //     for(int j=0; j < bucket_sizes[i]; ++j) std::cout << nodes_buckets[i*max_child+j] << " ";
        //     // for(int j=0; j < max_child; ++j) std::cout << nodes_buckets[i*max_child+j] << " ";
        //     std::cout << std::endl;
        // }
    }
    
    
    
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
    
    device_nodes_buckets.clear();
    device_nodes_buckets.shrink_to_fit();
    device_bucket_sizes.clear();
    device_bucket_sizes.shrink_to_fit();
    device_max_leaf_size.clear();
    device_max_leaf_size.shrink_to_fit();
    device_min_leaf_size.clear();
    device_min_leaf_size.shrink_to_fit();
    device_total_leaves.clear();
    device_total_leaves.shrink_to_fit();
    
    for(depth=0; depth < reached_max_depth; ++depth){
        device_tree[depth]->clear();
        device_tree[depth]->shrink_to_fit();
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
    
    if(VERBOSE >= 3){

        set<int> s; 
    
        for(int b=0; b < total_leaves; ++b){
            int count_cluster;
        
            count_cluster = bucket_sizes[b];
            if(count_cluster == 0) continue;

            std::vector<typepoints> X_axis(count_cluster);
            std::vector<typepoints> Y_axis(count_cluster);

            for(int i=0; i < count_cluster; ++i){
                int p = nodes_buckets[b*max_child+i];
                X_axis[i] = points[get_point_idx(p,0,N,D)];
                Y_axis[i] = points[get_point_idx(p,1,N,D)];
            }
            plt::scatter<typepoints,typepoints>(X_axis, Y_axis,1.0,{
                {"alpha", "0.8"}
            });

        }
        // plt::show();
        std::cerr << run_name << std::endl;

        plt::save(run_name);
    }

    end_tree_cron.stop();
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
        printf("KNN computation Kernel takes %lf seconds\n", cron_knn.t_total/1000);
        printf("End tree takes %lf seconds\n", end_tree_cron.t_total/1000);
    }
}


void RPFK::knn_gpu_rpfk_forest(int n_trees,
                               int K, int N, int D, int VERBOSE,
                               string run_name="tree")
{
    Cron forest_total_cron;
    forest_total_cron.start();
    thrust::device_vector<typepoints> device_points(points, points+N*D);
    thrust::device_vector<int> device_old_knn_indices(knn_indices, knn_indices+N*K);
    thrust::device_vector<int> device_knn_indices(knn_indices, knn_indices+N*K);
    thrust::device_vector<typepoints> device_knn_sqr_distances(knn_sqr_distances, knn_sqr_distances+N*K);

    for(int i=0; i < n_trees; ++i){
        add_random_projection_tree(device_points,
                     device_old_knn_indices,
                     device_knn_indices,
                     device_knn_sqr_distances,
                     K, N, D, VERBOSE-1,
                     run_name+"_"+std::to_string(i)+".png");
        RANDOM_SEED++;
    }

    forest_total_cron.stop();
    if(VERBOSE >= 1){
        printf("Creating RPFK forest takes %lf seconds\n", forest_total_cron.t_total/1000);
    }

    
    
    Cron cron_nearest_neighbors_exploring;
    cron_nearest_neighbors_exploring.start();

    // TODO: Automatically specify another id of GPU device rather than 0
    // Automatically get the ideal number of threads per block and total of blocks
    // used in kernels
    int devUsed = 0;
    cudaSetDevice(devUsed);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, devUsed);
    const int NT = deviceProp.maxThreadsPerBlock;
    const int NB = deviceProp.multiProcessorCount*(deviceProp.maxThreadsPerMultiProcessor/deviceProp.maxThreadsPerBlock);

    if(VERBOSE >= 2 && nn_exploring_factor > 0){
        for(int i=0; i < 80*4; ++i) std::cout<<" ";
        std::cout << std::endl;
        std::cout << "\e[ANearest Neighbor Exploring: " << "0/" << nn_exploring_factor << std::endl;
    }
    for(int i=0; i < nn_exploring_factor; ++i){
        thrust::copy(device_knn_indices.begin(), device_knn_indices.begin() + K*N, device_old_knn_indices.begin());
        cudaDeviceSynchronize();
        // nearest_neighbors_exploring<<<NB,NT>>>(thrust::raw_pointer_cast(device_points.data()),
        //                                        thrust::raw_pointer_cast(device_old_knn_indices.data()),
        //                                        thrust::raw_pointer_cast(device_knn_indices.data()),
        //                                        thrust::raw_pointer_cast(device_knn_sqr_distances.data()),
        //                                        N, D, K);
        nearest_neighbors_exploring_coalesced<<<NB,NT>>>(thrust::raw_pointer_cast(device_points.data()),
                                                         thrust::raw_pointer_cast(device_old_knn_indices.data()),
                                                         thrust::raw_pointer_cast(device_knn_indices.data()),
                                                         thrust::raw_pointer_cast(device_knn_sqr_distances.data()),
                                                         N, D, K);
        CudaTest((char *)"nearest_neighbors_exploring Kernel failed!");
        if(VERBOSE >= 2) std::cout << "\e[ANearest Neighbor Exploring: " << (i+1) << "/" << nn_exploring_factor << std::endl;
        cudaDeviceSynchronize();
    }
    cron_nearest_neighbors_exploring.stop();
    if(VERBOSE >= 1){
        printf("Nearest Neighbors Exploring computation Kernel takes %lf seconds\n", cron_nearest_neighbors_exploring.t_total/1000);
    }

    thrust::copy(device_knn_indices.begin(), device_knn_indices.begin() + N*K, knn_indices);
    thrust::copy(device_knn_sqr_distances.begin(), device_knn_sqr_distances.begin() + N*K, knn_sqr_distances);
    cudaDeviceSynchronize();

    device_points.clear();
    device_points.shrink_to_fit();
    device_knn_indices.clear();
    device_knn_indices.shrink_to_fit();
    device_old_knn_indices.clear();
    device_old_knn_indices.shrink_to_fit();
    device_knn_sqr_distances.clear();
    device_knn_sqr_distances.shrink_to_fit();

    
}

int main(int argc,char* argv[]) {
    // test_random<<<1,1>>>();
    // test_dynamic_vec_reg<<<1,1>>>(15);
    // thrust::device_vector<int> test_var(sizeof(int), 0);
    // test_atomic<<<1,1>>>(thrust::raw_pointer_cast(test_var.data()),15);
    
    
    // thrust::device_vector<int>* device_test;
    // device_test = new thrust::device_vector<int>(10);
    // test_cuda_dynamic_declaration<<<1,1>>>(thrust::raw_pointer_cast(device_test[0]->data()), 10);
    // cudaDeviceSynchronize();
    // thrust::copy(device_test->begin(), device_test->end(), std::ostream_iterator<int>(std::cout, "\n"));

    // cudaDeviceSynchronize();
    // return 0;

    // srand (time(NULL));
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


    
    typepoints* points = (typepoints*) malloc(sizeof(typepoints)*N*D);
    int K = 32;

    int* knn_indices = (int*) malloc(sizeof(int)*N*K);
    std::fill_n(knn_indices, N*K, -1);

    typepoints* knn_sqr_distances = (typepoints*) malloc(sizeof(typepoints)*N*K);
    std::fill_n(knn_sqr_distances, N*K, FLT_MAX);


    std::vector<int> labels(N*D);

    int l;
    string type_init;
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
    RPFK rpfk_knn(points, knn_indices, knn_sqr_distances, K, MAX_DEPTH, RANDOM_SEED, nn_exploring_factor);
    rpfk_knn.knn_gpu_rpfk_forest(5, K, N, D, VERBOSE, "tree");


}
