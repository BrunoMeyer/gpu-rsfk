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


#include <curand.h>
#include <curand_kernel.h>

#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;
 

#include <iostream> 
#include <stdio.h>
using namespace std; 
#include <cstdlib>
#include <cmath>

#define HEAP_PARENT(i) ((i-1)/2)
#define HEAP_LEFT(i) ((2*i)+1)
#define HEAP_RIGHT(i) ((2*i)+2)

// #include <boost/preprocessor/repetition/repeat.hpp>


/*
PQ = Q-P

# Find equation hiperplane formula
H_bias = PQ.dot(Q)
H = PQ

# Verify if P is in the upper side of hiperplane
P_side = P.dot(H) > H_bias
*/


#define typepoints float


typedef struct TreeNode {
    // // A normal vector and a point of the hyperplane is sufficiently to represent it
    float* hyperplane_normal; // Difference between two points
    float* hyperplane_point; // Average of two points used to create a node
    
    // int skip_link;
    TreeNode* negative_path;
    TreeNode* positive_path;

    int leafs_count;
} TreeNode;


__device__
inline
void init_TreeNode(TreeNode* n, int node_index, TreeNode* tree, unsigned int* points_path, typepoints* points, int* actual_depth, int N, int D){
    n->leafs_count = 0;
    n->negative_path = nullptr;
    n->positive_path = nullptr;
    // n->negative_path = hyperplane_normal;
    // n->positive_path = hyperplane_point;
}

// __device__
// inline
// void init_TreeNode(TreeNode* n, int p1, int p2, int N, int D, typepoints* points){

// }


#define RANDOM_SEED 0


__global__
void test_random(){
    curandState_t r; 
    curand_init(RANDOM_SEED, /* the seed controls the sequence of random values that are produced */
            blockIdx.x, /* the sequence number is only important with multiple cores */
            threadIdx.x, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
            //   &states[blockIdx.x]);
            &r);
    printf("%d\n",curand(&r) % 100);
    printf("%d\n",curand(&r) % 100);
    printf("%d\n",curand(&r) % 100);
}

__global__
void test(typepoints* arr, int N, int D){
    curandState_t r; 
    curand_init(RANDOM_SEED, /* the seed controls the sequence of random values that are produced */
              blockIdx.x, /* the sequence number is only important with multiple cores */
              threadIdx.x, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
            //   &states[blockIdx.x]);
              &r);

    // arr[threadIdx.x*N+1] = N/2 + N/4;
    arr[threadIdx.x*D+1] = curand(&r) % N;
}

__device__
// inline
void tree_search(int point, TreeNode* node, TreeNode* tree, unsigned int* points_path, typepoints* points, int* actual_depth, int N, int D){
    
}

__device__
inline
void create_node(unsigned int node_idx, int p1, int p2, typepoints* tree, typepoints* points, int D)
{
    // Average point
    // node_path*D*2 : D*2 = size of centroid point and normal vector

    unsigned int i;
    for(i=0; i < D; ++i){
        tree[node_idx*(2*D+1) + i] = (points[p1*D+i]+points[p2*D+i])/2;
    }
    

    register typepoints normal_vector_axis_val;
    register typepoints plane_bias = 0;
    for(i=0;i < D; ++i){
        normal_vector_axis_val = points[p2*D+i]-points[p1*D+i];
        tree[node_idx*(2*D+1) + i + D] = normal_vector_axis_val;
        plane_bias+= normal_vector_axis_val*tree[node_idx*(2*D+1) + i]; // multiply the point of plane and the normal vector 

    }
    tree[node_idx*(2*D+1) + 2*D] = plane_bias;

    // printf("#########\n");
    // for(i=0; i < D; ++i){
    //     printf("%f ", points[p1*D + i]);
    // }
    // printf("\n");
    // for(i=0; i < D; ++i){
    //     printf("%f ", points[p2*D + i]);
    // }
    // printf("\n");
    // for(i=0; i < D; ++i){
    //     printf("%f ", tree[node_idx*(2*D+1) + i]);
    // }
    // printf("\n");
    // for(i=0; i < D; ++i){
    //     printf("%f ", tree[node_idx*(2*D+1) + i + D]);
    // }
    // printf("\n");
    // printf("%f\n", tree[node_idx*(2*D+1) + 2*D]);
    // printf("#########\n");

}


__device__
inline
int check_hyperplane_side(unsigned int node_idx, int p, typepoints* tree, typepoints* points, int D)
{
    int i;
    float aux = 0;
    for(i=0; i < D; ++i){
        aux += tree[node_idx*(2*D+1) + i+D]*points[D*p + i];
    }

    return aux > tree[node_idx*(2*D+1) + 2*D];
}


__global__
void build_tree(typepoints* tree, unsigned int* points_parent, typepoints* points, int* actual_depth, int N, int D){
    curandState_t r; 
    curand_init(RANDOM_SEED, /* the seed controls the sequence of random values that are produced */
            blockIdx.x, /* the sequence number is only important with multiple cores */
            threadIdx.x, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
            //   &states[blockIdx.x]);
            &r);


    int p1, p2;

    // Sample two random points

    // int nodes_to_process = 1; // Root
    // int depth = 0;
    unsigned int node_thread = 0;
    // unsigned int node_idx = 0;
    int p;
    // unsigned int mask_depth = 2; // Start in second depth (second bit)
    
    

    if(threadIdx.x == 0){
        p1 = curand(&r) % N;
        p2 = p1;
        // Ensure that two different points was sampled
        while(p1 == p2 && N > 1){
            p2 = curand(&r) % N;
        }
        printf("Sampled points: %d %d\n", p1,p2);

        create_node(0, p1, p2, tree, points, D);
        *actual_depth = 1;
    }
    
    __syncthreads();
    
    

    while(*actual_depth < 10){
        
        // Set nodes parent in the new depth
        for(p = threadIdx.x; p < N; p+=blockDim.x){
            // Heap left and right nodes are separeted by 1
            int right_child = check_hyperplane_side(points_parent[p], p, tree, points, D);
            points_parent[p] =  HEAP_LEFT(points_parent[p])+right_child;
        }
        __syncthreads();
        
        
        
        // Create new nodes
        if(threadIdx.x < pow(2,*actual_depth)){
            node_thread = 0; // start on root
            unsigned int bit_mask = 1;
            // Each thread find the node index to be created
            for(unsigned int i=0; i < *actual_depth; ++i){
                node_thread = HEAP_LEFT(node_thread) + ((i & bit_mask) != 0);
                bit_mask = pow(2,*actual_depth);

            }
            
            // if(threadIdx.x==0){
            //     for(unsigned int i=0; i < N; ++i){
            //         printf("%d\n", points_parent[i]);
            //     }
            // }
            // return;
            p1 = curand(&r) % N;
            int init_p_search = p1;
            while(points_parent[p1] != node_thread){
                p1=((p1+1)%N);
                if(p1 == init_p_search){
                    printf("AAAAAAAAAAAAAAA\n");
                    break;
                }
                // printf("%d\n",p1);
            }
            p2 = curand(&r) % N;
            // Ensure that two different points was sampled
            // init_p_search = p2;
            while(p1 == p2  &&  points_parent[p2] != node_thread){
                p2=((p1+1)%N);
                if(p2 == init_p_search){
                    printf("BBBBBBBBBBBBBBBB\n");
                    break;
                }
                // printf("%d\n",p2);

            }
            

            
            create_node(node_thread, p1, p2, tree, points, D);
        }

        __syncthreads();

        if(threadIdx.x == 0){
            *actual_depth = *actual_depth+1;
            // printf("%d\n", *actual_depth);
        }
        __syncthreads();
    }

    // if(threadIdx.x==0){
    //     for(unsigned int i=0; i < N; ++i){
    //         printf("%d\n", points_parent[i]);
    //     }
    // }

    
    return;
}

int main(int argc,char* argv[]) {
    // test_random<<<1,1>>>();
    // cudaDeviceSynchronize();
    // return 0;

    int N = atoi(argv[1]);
    int D = atoi(argv[2]);
    std::cout << N << std::endl;
    std::cout << D << std::endl;
    std::cout << std::endl;

    // thrust::copy(knn_indices, knn_indices + num_points * num_neighbors, knn_indices_device.begin());
    // thrust::device_vector<long> knn_indices_long_device(knn_indices_long, knn_indices_long + num_points * num_neighbors);   
    
    std::vector<typepoints> points(N*D);
    for(int i=0; i < N; ++i){
        for(int j=0; j < D; ++j){
            points[i*D+j] = i + (i>N/2)*N/2;
        }
    }


    std::vector<typepoints> X_axis(N);
    std::vector<typepoints> Y_axis(N);
    
    for(int i=0; i < N; ++i){
        X_axis[i] = points[i*D];
        Y_axis[i] = points[i*D + 1];
    }
    // plt::scatter<typepoints,typepoints>(X_axis, Y_axis,10.0);
    // plt::show();

    thrust::device_vector<typepoints> device_points(points.begin(), points.end());
    // test<<<1,100>>>(thrust::raw_pointer_cast(device_points.data()), N, D);

    int MAX_NODES = 2*N-1 - N;
    std::cout << "MAX NODES: " << MAX_NODES << std::endl;
    // thrust::device_vector<TreeNode> device_tree(sizeof(TreeNode) * (MAX_NODES));
    thrust::device_vector<typepoints> device_tree((2*D + 1)*sizeof(typepoints) * (MAX_NODES));
    thrust::device_vector<int> actual_depth(sizeof(int) * 1);
    thrust::device_vector<unsigned int> points_parent(sizeof(unsigned int) * N, 0);

    const int nt = 1024;
    build_tree<<<1,nt>>>(thrust::raw_pointer_cast(device_tree.data()),
                        thrust::raw_pointer_cast(points_parent.data()),
                        thrust::raw_pointer_cast(device_points.data()),
                        thrust::raw_pointer_cast(actual_depth.data()),
                        N, D);
    

    thrust::copy(device_points.begin(), device_points.begin() + N*D, points.begin());
    for(int i=0; i < N; ++i){
        X_axis[i] = points[i*D];
        Y_axis[i] = points[i*D + 1];
    }
    // plt::scatter<typepoints,typepoints>(X_axis, Y_axis,10.0);
    // plt::show();
}
