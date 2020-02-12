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

struct is_true
{
  __host__ __device__
  bool operator()(const bool x)
  {
    return x;
  }
};

struct is_greater_than_zero
{
  __host__ __device__
  bool operator()(const int x)
  {
    return x > 0;
  }
};


#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;

#include <iostream> 
#include <stdio.h>
using namespace std; 
#include <cstdlib>
#include <cmath>
#include <bits/stdc++.h> 

#define typepoints float
// #define MAX_DEPTH 11
// #define RANDOM_SEED 42
#define RANDOM_SEED 0
#define MAX_K 1024
#define MAX_TREE_CHILD 128

#define HEAP_PARENT(i) ((i-1)/2)
#define HEAP_LEFT(i) ((2*i)+1)
#define HEAP_RIGHT(i) ((2*i)+2)

#include "random_tree.cu"
#include "kernels/build_tree_init.cu"
#include "kernels/build_tree_check_points_side.cu"
#include "kernels/build_tree_count_new_nodes.cu"
#include "kernels/build_tree_create_nodes.cu"
#include "kernels/build_tree_update_parents.cu"
#include "kernels/build_tree_utils.cu"
#include "kernels/build_tree_bucket_points.cu"
#include "kernels/compute_knn_from_buckets.cu"



static void CudaTest(char* msg)
{
  cudaError_t e;

  cudaThreadSynchronize();
  if (cudaSuccess != (e = cudaGetLastError())) {
    fprintf(stderr, "%s: %d\n", msg, e);
    fprintf(stderr, "%s\n", cudaGetErrorString(e));
    exit(1);
  }
}



int countDistinct(int* arr, int n) 
{ 
    // Creates an empty hashset 
    unordered_set<int> s; 
  
    // Traverse the input array 
    int res = 0; 
    for (int i = 0; i < n; i++) { 
  
        // If not present, then put it in 
        // hashtable and increment result 
        if (s.find(arr[i]) == s.end()) { 
            s.insert(arr[i]); 
            res++; 
        } 
    } 
  
    return res; 
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

class RPTK
{
public:
    std::vector<typepoints> &points;
    std::vector<int> &knn_indices;
    std::vector<typepoints> &knn_sqr_distances;
    
    RPTK(std::vector<typepoints> &points,
         std::vector<int> &knn_indices,
         std::vector<typepoints> &knn_sqr_distances):
         points(points),
         knn_indices(knn_indices),
         knn_sqr_distances(knn_sqr_distances){}
    
    void knn_gpu_rptk(thrust::device_vector<typepoints> &device_points,
                      thrust::device_vector<int> &device_knn_indices,
                      thrust::device_vector<typepoints> &device_knn_sqr_distances,
                      int K, int N, int D, int MAX_DEPTH, int VERBOSE,
                      string run_name);
    
    void knn_gpu_rptk_forest(int n_trees,
                             int K, int N, int D, int MAX_DEPTH, int VERBOSE,
                             string run_name);
};

void RPTK::knn_gpu_rptk(thrust::device_vector<typepoints> &device_points,
                        thrust::device_vector<int> &device_knn_indices,
                        thrust::device_vector<typepoints> &device_knn_sqr_distances,
                        int K, int N, int D, int MAX_DEPTH, int VERBOSE,
                        string run_name="out.png")
{
    thrust::device_vector<typepoints>** device_tree = (thrust::device_vector<typepoints>**) malloc(sizeof(thrust::device_vector<typepoints>*)*MAX_DEPTH);
    thrust::device_vector<int>** device_tree_parents = (thrust::device_vector<int>**) malloc(sizeof(thrust::device_vector<int>*)*MAX_DEPTH);
    thrust::device_vector<int>** device_tree_children = (thrust::device_vector<int>**) malloc(sizeof(thrust::device_vector<int>*)*MAX_DEPTH);
    thrust::device_vector<bool>** device_is_leaf = (thrust::device_vector<bool>**) malloc(sizeof(thrust::device_vector<bool>*)*MAX_DEPTH);
    thrust::device_vector<int>** device_sample_points = (thrust::device_vector<int>**) malloc(sizeof(thrust::device_vector<int>*)*MAX_DEPTH);
    thrust::device_vector<int>** device_child_count = (thrust::device_vector<int>**) malloc(sizeof(thrust::device_vector<int>*)*MAX_DEPTH);
    thrust::device_vector<typepoints>** device_tree_tmp = (thrust::device_vector<typepoints>**) malloc(sizeof(thrust::device_vector<typepoints>*)*MAX_DEPTH);
    thrust::device_vector<int>** device_tree_parents_tmp = (thrust::device_vector<int>**) malloc(sizeof(thrust::device_vector<int>*)*MAX_DEPTH);
    thrust::device_vector<int>** device_tree_children_tmp = (thrust::device_vector<int>**) malloc(sizeof(thrust::device_vector<int>*)*MAX_DEPTH);
    thrust::device_vector<bool>** device_is_leaf_tmp = (thrust::device_vector<bool>**) malloc(sizeof(thrust::device_vector<bool>*)*MAX_DEPTH);
    thrust::device_vector<int>** device_sample_points_tmp = (thrust::device_vector<int>**) malloc(sizeof(thrust::device_vector<int>*)*MAX_DEPTH);
    thrust::device_vector<int>** device_child_count_tmp = (thrust::device_vector<int>**) malloc(sizeof(thrust::device_vector<int>*)*MAX_DEPTH);
    

    int MAX_NODES = 1;
    device_tree[0] = new thrust::device_vector<typepoints>(((D+1)*MAX_NODES));
    device_tree_parents[0] = new thrust::device_vector<int>(MAX_NODES,-1);
    device_tree_children[0] = new thrust::device_vector<int>(2*MAX_NODES,-1);
    device_is_leaf[0] = new thrust::device_vector<bool>(MAX_NODES, false);
    device_sample_points[0] = new thrust::device_vector<int>(4*MAX_NODES, -1);
    device_child_count[0] = new thrust::device_vector<int>(MAX_NODES, 0);

    thrust::device_vector<int> device_points_parent(N, 0);
    thrust::device_vector<int> device_points_depth(N, 0);
    thrust::device_vector<int> device_is_right_child(N, 0);

    thrust::device_vector<int> device_depth_level_count(MAX_DEPTH,-1);
    thrust::device_vector<int> device_accumulated_nodes_count(MAX_DEPTH,-1);
    thrust::device_vector<int> device_tree_count(1,0);
    thrust::device_vector<int> device_count_new_nodes(1,0);
    thrust::device_vector<int> device_actual_depth(1,0);

    const int nt = 1024;
    const int mp = 8;


    build_tree_init<<<mp,nt>>>(thrust::raw_pointer_cast(device_tree[0]->data()),
                               thrust::raw_pointer_cast(device_tree_parents[0]->data()),
                               thrust::raw_pointer_cast(device_tree_children[0]->data()),
                               thrust::raw_pointer_cast(device_points_parent.data()),
                               thrust::raw_pointer_cast(device_child_count[0]->data()),
                               thrust::raw_pointer_cast(device_is_leaf[0]->data()),
                               thrust::raw_pointer_cast(device_points.data()),
                               thrust::raw_pointer_cast(device_actual_depth.data()),
                               thrust::raw_pointer_cast(device_tree_count.data()),
                               thrust::raw_pointer_cast(device_depth_level_count.data()),
                               N, D);
    cudaDeviceSynchronize();
    CudaTest((char *)"build_tree_init Kernel failed!");

    if(VERBOSE >= 1){
        std::cout << std::endl;
    }

    Cron total_cron, update_parents_cron, create_nodes_cron,
         check_points_side_cron, tree_count_cron;

    total_cron.start();
    int depth, count_new_nodes, count_total_nodes;
    
    count_total_nodes = 1;
    
    for(depth=1; depth < MAX_DEPTH; ++depth){
        cudaDeviceSynchronize();
        
        check_points_side_cron.start();
        build_tree_check_points_side<<<mp,nt>>>(thrust::raw_pointer_cast(device_tree[depth-1]->data()),
                                   thrust::raw_pointer_cast(device_tree_parents[depth-1]->data()),
                                   thrust::raw_pointer_cast(device_tree_children[depth-1]->data()),
                                   thrust::raw_pointer_cast(device_points_parent.data()),
                                   thrust::raw_pointer_cast(device_points_depth.data()),
                                   thrust::raw_pointer_cast(device_is_right_child.data()),
                                   thrust::raw_pointer_cast(device_is_leaf[depth-1]->data()),
                                   thrust::raw_pointer_cast(device_sample_points[depth-1]->data()),
                                   thrust::raw_pointer_cast(device_child_count[depth-1]->data()),
                                   thrust::raw_pointer_cast(device_points.data()),
                                   thrust::raw_pointer_cast(device_actual_depth.data()),
                                   thrust::raw_pointer_cast(device_tree_count.data()),
                                   thrust::raw_pointer_cast(device_depth_level_count.data()),
                                   N, D);
        cudaDeviceSynchronize();
        check_points_side_cron.stop();
        CudaTest((char *)"build_tree_check_points_side Kernel failed!");
        

         tree_count_cron.start();
        build_tree_count_new_nodes<<<mp,nt>>>(thrust::raw_pointer_cast(device_tree[depth-1]->data()),
                                   thrust::raw_pointer_cast(device_tree_parents[depth-1]->data()),
                                   thrust::raw_pointer_cast(device_tree_children[depth-1]->data()),
                                   thrust::raw_pointer_cast(device_points_parent.data()),
                                   thrust::raw_pointer_cast(device_is_right_child.data()),
                                   thrust::raw_pointer_cast(device_is_leaf[depth-1]->data()),
                                   thrust::raw_pointer_cast(device_sample_points[depth-1]->data()),
                                   thrust::raw_pointer_cast(device_child_count[depth-1]->data()),
                                   thrust::raw_pointer_cast(device_points.data()),
                                   thrust::raw_pointer_cast(device_actual_depth.data()),
                                   thrust::raw_pointer_cast(device_tree_count.data()),
                                   thrust::raw_pointer_cast(device_depth_level_count.data()),
                                   thrust::raw_pointer_cast(device_count_new_nodes.data()),
                                   N, D);
        cudaDeviceSynchronize();
        tree_count_cron.stop();
        CudaTest((char *)"build_tree_count_new_nodes Kernel failed!");
        
        
        thrust::copy(device_count_new_nodes.begin(), device_count_new_nodes.begin()+1, &count_new_nodes);
        cudaDeviceSynchronize();
        // std::cout << "Add " << count_new_nodes << " new nodes" << std::endl;
        count_total_nodes+=count_new_nodes;

        device_tree[depth] = new thrust::device_vector<typepoints>(((D+1)*count_new_nodes));
        device_tree_parents[depth] = new thrust::device_vector<int>(count_new_nodes,-1);
        device_tree_children[depth] = new thrust::device_vector<int>(2*count_new_nodes,-1);
        device_is_leaf[depth] = new thrust::device_vector<bool>(count_new_nodes, false);
        device_sample_points[depth] = new thrust::device_vector<int>(4*count_new_nodes, -1);
        device_child_count[depth] = new thrust::device_vector<int>(count_new_nodes, 0);

        
        
        


        create_nodes_cron.start();
        build_tree_create_nodes<<<mp,nt>>>(thrust::raw_pointer_cast(device_tree[depth]->data()), // new depth is created
                                           thrust::raw_pointer_cast(device_tree_parents[depth]->data()), // new depth is created
                                           thrust::raw_pointer_cast(device_tree_children[depth-1]->data()),
                                           thrust::raw_pointer_cast(device_points_parent.data()),
                                           thrust::raw_pointer_cast(device_is_right_child.data()),
                                           thrust::raw_pointer_cast(device_is_leaf[depth-1]->data()),
                                           thrust::raw_pointer_cast(device_is_leaf[depth]->data()), // new depth is created
                                           thrust::raw_pointer_cast(device_sample_points[depth-1]->data()),
                                           thrust::raw_pointer_cast(device_child_count[depth-1]->data()),
                                           thrust::raw_pointer_cast(device_points.data()),
                                           thrust::raw_pointer_cast(device_actual_depth.data()),
                                           thrust::raw_pointer_cast(device_tree_count.data()),
                                           thrust::raw_pointer_cast(device_depth_level_count.data()),
                                           thrust::raw_pointer_cast(device_count_new_nodes.data()),
                                           N, D);
        cudaDeviceSynchronize();
        create_nodes_cron.stop();
        CudaTest((char *)"build_tree_create_nodes Kernel failed!");


        update_parents_cron.start();
        build_tree_update_parents<<<mp,nt>>>(thrust::raw_pointer_cast(device_tree[depth-1]->data()),
                                             thrust::raw_pointer_cast(device_tree_parents[depth-1]->data()),
                                             thrust::raw_pointer_cast(device_tree_children[depth-1]->data()),
                                             thrust::raw_pointer_cast(device_points_parent.data()),
                                             thrust::raw_pointer_cast(device_points_depth.data()),
                                             thrust::raw_pointer_cast(device_is_right_child.data()),
                                             thrust::raw_pointer_cast(device_is_leaf[depth-1]->data()),
                                             thrust::raw_pointer_cast(device_sample_points[depth-1]->data()),
                                             thrust::raw_pointer_cast(device_child_count[depth-1]->data()),
                                             thrust::raw_pointer_cast(device_child_count[depth]->data()), // new depth is created
                                             thrust::raw_pointer_cast(device_points.data()),
                                             thrust::raw_pointer_cast(device_actual_depth.data()),
                                             thrust::raw_pointer_cast(device_tree_count.data()),
                                             thrust::raw_pointer_cast(device_depth_level_count.data()),
                                             thrust::raw_pointer_cast(device_count_new_nodes.data()),
                                             N, D);
        cudaDeviceSynchronize();
        CudaTest((char *)"build_tree_update_parents Kernel failed!");
        build_tree_post_update_parents<<<mp,nt>>>(thrust::raw_pointer_cast(device_tree[depth-1]->data()),
                                                  thrust::raw_pointer_cast(device_tree_parents[depth-1]->data()),
                                                  thrust::raw_pointer_cast(device_tree_children[depth-1]->data()),
                                                  thrust::raw_pointer_cast(device_points_parent.data()),
                                                  thrust::raw_pointer_cast(device_points_depth.data()),
                                                  thrust::raw_pointer_cast(device_is_right_child.data()),
                                                  thrust::raw_pointer_cast(device_is_leaf[depth-1]->data()),
                                                  thrust::raw_pointer_cast(device_sample_points[depth-1]->data()),
                                                  thrust::raw_pointer_cast(device_child_count[depth-1]->data()),
                                                  thrust::raw_pointer_cast(device_points.data()),
                                                  thrust::raw_pointer_cast(device_actual_depth.data()),
                                                  thrust::raw_pointer_cast(device_tree_count.data()),
                                                  thrust::raw_pointer_cast(device_depth_level_count.data()),
                                                  thrust::raw_pointer_cast(device_count_new_nodes.data()),
                                                  N, D);
        cudaDeviceSynchronize();
        update_parents_cron.stop();
        CudaTest((char *)"build_tree_post_update_parents Kernel failed!");


        build_tree_utils<<<1,1>>>(thrust::raw_pointer_cast(device_actual_depth.data()),
                                  thrust::raw_pointer_cast(device_depth_level_count.data()),
                                  thrust::raw_pointer_cast(device_count_new_nodes.data()),
                                  thrust::raw_pointer_cast(device_tree_count.data()));
        cudaDeviceSynchronize();
        CudaTest((char *)"build_tree_utils Kernel failed!");

        if(VERBOSE >= 1){
            std::cout << "\e[ABuilding Tree Depth: " << depth+1 << "/" << MAX_DEPTH;
            std::cout << " | Total nodes: " << count_total_nodes << std::endl;
        }
    }
    build_tree_fix<<<1,1>>>(thrust::raw_pointer_cast(device_depth_level_count.data()),
                            thrust::raw_pointer_cast(device_tree_count.data()),
                            thrust::raw_pointer_cast(device_accumulated_nodes_count.data()),
                            MAX_DEPTH);
    cudaDeviceSynchronize();
    CudaTest((char *)"build_tree_fix Kernel failed!");
    
    total_cron.stop();
    
    Cron cron_classify_points;
    cron_classify_points.start();

    thrust::device_vector<int> device_max_leaf_size(1,0);
    thrust::device_vector<int> device_total_leafs(1,0);
    
    for(depth=0; depth < MAX_DEPTH; ++depth){
        build_tree_max_leaf_size<<<mp,nt>>>(thrust::raw_pointer_cast(device_max_leaf_size.data()),
                                            thrust::raw_pointer_cast(device_total_leafs.data()),
                                            thrust::raw_pointer_cast(device_is_leaf[depth]->data()),
                                            thrust::raw_pointer_cast(device_child_count[depth]->data()),
                                            thrust::raw_pointer_cast(device_tree_count.data()),
                                            thrust::raw_pointer_cast(device_depth_level_count.data()),
                                            depth);
        cudaDeviceSynchronize();
        CudaTest((char *)"build_tree_max_leaf_size Kernel failed!");
    }

    
    int max_child, total_leafs;
    thrust::copy(device_max_leaf_size.begin(), device_max_leaf_size.begin()+1, &max_child);
    thrust::copy(device_total_leafs.begin(), device_total_leafs.begin()+1, &total_leafs);
    cudaDeviceSynchronize();

    if(VERBOSE >= 1){
        std::cout << "Max child count: " << max_child << std::endl;
        std::cout << "Total leafs: " << total_leafs << std::endl;
    }




    thrust::copy(device_tree_count.begin(), device_tree_count.begin()+1, &MAX_NODES);
    
    thrust::device_vector<int> device_nodes_buckets(MAX_NODES*max_child, -1);
    thrust::device_vector<int> device_count_buckets(MAX_NODES, 0);
    build_tree_bucket_points<<<mp,nt>>>(thrust::raw_pointer_cast(device_points_parent.data()),
                                        thrust::raw_pointer_cast(device_points_depth.data()),
                                        thrust::raw_pointer_cast(device_accumulated_nodes_count.data()),
                                        thrust::raw_pointer_cast(device_nodes_buckets.data()),
                                        thrust::raw_pointer_cast(device_count_buckets.data()),
                                        N, max_child);
    cudaDeviceSynchronize();
    cron_classify_points.stop();
    CudaTest((char *)"build_tree_bucket_points Kernel failed!");
    std::vector<int> nodes_buckets(MAX_NODES*max_child);
    thrust::copy(device_nodes_buckets.begin(), device_nodes_buckets.begin() + MAX_NODES*max_child, nodes_buckets.begin());
    std::vector<int> count_buckets(MAX_NODES);
    thrust::copy(device_count_buckets.begin(), device_count_buckets.begin() + MAX_NODES, count_buckets.begin());
    cudaDeviceSynchronize();

    
    
    


    Cron cron_knn;
    cron_knn.start();

    compute_knn_from_buckets<<<mp,nt>>>(thrust::raw_pointer_cast(device_points_parent.data()),
                                        thrust::raw_pointer_cast(device_points_depth.data()),
                                        thrust::raw_pointer_cast(device_accumulated_nodes_count.data()),
                                        thrust::raw_pointer_cast(device_count_buckets.data()),
                                        thrust::raw_pointer_cast(device_points.data()),
                                        thrust::raw_pointer_cast(device_nodes_buckets.data()),
                                        thrust::raw_pointer_cast(device_knn_indices.data()),
                                        thrust::raw_pointer_cast(device_knn_sqr_distances.data()),
                                        N, D, max_child, K);
    cudaDeviceSynchronize();
    cron_knn.stop();    
    CudaTest((char *)"compute_knn_from_buckets Kernel failed!");

    thrust::copy(device_knn_indices.begin(), device_knn_indices.begin() + N*K, knn_indices.begin());
    thrust::copy(device_knn_sqr_distances.begin(), device_knn_sqr_distances.begin() + N*K, knn_sqr_distances.begin());
    cudaDeviceSynchronize();

    if(VERBOSE >= 2){
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
        //     // std::cout << i << "(" << count_buckets[i] << ") : ";
        //     for(int j=0; j < count_buckets[i]; ++j) std::cout << nodes_buckets[i*max_child+j] << " ";
        //     // for(int j=0; j < MAX_TREE_CHILD; ++j) std::cout << nodes_buckets[i*MAX_TREE_CHILD+j] << " ";
        //     std::cout << std::endl;
        // }
    }
    




    if(VERBOSE >= 1){
        printf("Total tree building takes %lf seconds\n", total_cron.t_total/1000);
        printf("Check points side Kernel takes %lf seconds\n", check_points_side_cron.t_total/1000);
        printf("Count new nodes Kernel takes %lf seconds\n", tree_count_cron.t_total/1000);
        printf("Create nodes Kernel takes %lf seconds\n", create_nodes_cron.t_total/1000);
        printf("Update parents Kernel takes %lf seconds\n", update_parents_cron.t_total/1000);
        printf("Create nodes Kernel takes %lf seconds\n", create_nodes_cron.t_total/1000);
        printf("Bucket creation Kernel takes %lf seconds\n", cron_classify_points.t_total/1000);
        printf("KNN computation Kernel takes %lf seconds\n", cron_knn.t_total/1000);
    }

    if(VERBOSE >= 3){

        set<int> s; 
    
        for(int b=0; b < MAX_NODES; ++b){
            int count_cluster;
        
            count_cluster = count_buckets[b];
            if(count_cluster == 0) continue;

            std::vector<typepoints> X_axis(count_cluster);
            std::vector<typepoints> Y_axis(count_cluster);

            for(int i=0; i < count_cluster; ++i){
                int p = nodes_buckets[b*max_child+i];
                X_axis[i] = points[D*p];
                Y_axis[i] = points[D*p + 1];
            }
            plt::scatter<typepoints,typepoints>(X_axis, Y_axis,1.0,{
                {"alpha", "0.8"}
            });

        }
        // plt::show();
        plt::save(run_name);
    }
}


void RPTK::knn_gpu_rptk_forest(int n_trees,
                               int K, int N, int D, int MAX_DEPTH, int VERBOSE,
                               string run_name="out")
{
    thrust::device_vector<typepoints> device_points(points.begin(), points.end());
    thrust::device_vector<int> device_knn_indices(knn_indices.begin(), knn_indices.end());
    thrust::device_vector<typepoints> device_knn_sqr_distances(knn_sqr_distances.begin(), knn_sqr_distances.end());

    Cron forest_total_cron;
    forest_total_cron.start();
    for(int i=0; i < n_trees; ++i){
        knn_gpu_rptk(device_points,
                     device_knn_indices,
                     device_knn_sqr_distances,
                     K, N, D, MAX_DEPTH, VERBOSE-1,
                     run_name+"_"+std::to_string(i));
    }

    forest_total_cron.stop();
    if(VERBOSE >= 1){
        printf("Creating RPTK forest takes %lf seconds\n", forest_total_cron.t_total/1000);
    }
    
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
    srand(RANDOM_SEED);

    int N = atoi(argv[1]);
    int D = atoi(argv[2]);
    int MAX_DEPTH = atoi(argv[3]);
    int VERBOSE = atoi(argv[4]);
    int DS = atoi(argv[5]);


    
    std::vector<typepoints> points(N*D);
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
                points[i*D+j] = rand() % N;
            }
            if(type_init == "SPLITED_LINE"){
                points[i*D+j] = l + (l>N/2)*N/2;
            }
        }
        labels[i] = (l>N/2);
    }


    
    int K = 8;

    std::vector<int> knn_indices(N*K, -1);
    std::vector<typepoints> knn_sqr_distances(N*K, FLT_MAX);
    
    // knn_gpu_rptk(points, knn_indices, knn_sqr_distances, K, N, D, MAX_DEPTH, VERBOSE, "run1");
    // knn_gpu_rptk(points, knn_indices, knn_sqr_distances, K, N, D, MAX_DEPTH, VERBOSE, "run2");

    RPTK rptk_knn(points, knn_indices, knn_sqr_distances);
    rptk_knn.knn_gpu_rptk_forest(5, K, N, D, MAX_DEPTH, VERBOSE, "run");

}
