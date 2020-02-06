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

#include <curand.h>
#include <curand_kernel.h>

// #include <cooperative_groups.h>
// namespace cg = cooperative_groups;


#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;

#include <iostream> 
#include <stdio.h>
using namespace std; 
#include <cstdlib>
#include <cmath>
#include <bits/stdc++.h> 

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
// #define MAX_DEPTH 11
// #define RANDOM_SEED 42
#define RANDOM_SEED 0
#define MAX_K 1024
#define MAX_TREE_CHILD 64

typedef struct TreeNode
{
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
void init_TreeNode(TreeNode* n, int node_index, TreeNode* tree, unsigned int* points_path, typepoints* points, int* actual_depth, int N, int D)
{
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




__global__
void test_random()
{
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
void
test_dynamic_vec_reg(int s)
{
    // register int* arr = new int[s];
    register int arr[MAX_K];
    for(int i=0; i < s; ++i) arr[i] = i*10; 
    for(int i=0; i < s; ++i) printf("%d\n",arr[i]); 
}

__global__
void
test_atomic(int* v, int v2)
{
    // register int* arr = new int[s];
    int x = 0;
    *v = 42;
    // atomicAdd Returns the last value in stored in v before Add
    x = atomicAdd(v,(v2+1));
    printf("%d %d %d\n",x,*v,v2);

}


__global__
void test(typepoints* arr, int N, int D)
{
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
inline
void create_node(unsigned int node_idx, int p1, int p2, typepoints* tree, typepoints* points, int D)
{
    // Average point
    // node_path*D*2 : D*2 = size of centroid point and normal vector

    int i;
    // typepoints mean_axis_val;
    // typepoints normal_vector_axis_val;
    // typepoints plane_bias = 0;
    tree[node_idx*(D+1) + D] = 0.0f;

    for(i=0;i < D; ++i){
        // mean_axis_val = (points[p1*D+i]+points[p2*D+i])/2;
        // normal_vector_axis_val = points[p2*D+i]-points[p1*D+i];
        // tree[node_idx*(D+1) + i] = normal_vector_axis_val;
        // plane_bias+= tree[node_idx*(D+1) + i]*(points[p1*D+i]+points[p2*D+i])/2; // multiply the point of plane and the normal vector 
        tree[node_idx*(D+1) + i] = points[p1*D+i]-points[p2*D+i];
        tree[node_idx*(D+1) + D]+= tree[node_idx*(D+1) + i]*(points[p1*D+i]+points[p2*D+i])/2; // multiply the point of plane and the normal vector 
    }
    // tree[node_idx*(D+1) + D] = plane_bias;


    // if(node_idx == 2){
    //     printf("#########\n");
    //     for(i=0; i < D; ++i){
    //         printf("%f ", points[p1*D + i]);
    //     }
    //     printf("\n");
    //     for(i=0; i < D; ++i){
    //         printf("%f ", points[p2*D + i]);
    //     }
    //     printf("\n");
    //     for(i=0; i < D; ++i){
    //         printf("%f ", tree[node_idx*(D+1) + i]);
    //     }
    //     printf("\n");
    //     printf("%f\n", tree[node_idx*(D+1) + D]);
    //     printf("#########\n");
    // }
}


__device__
inline
int check_hyperplane_side(unsigned int node_idx, int p, typepoints* tree, typepoints* points, int D)
{
    typepoints aux = 0.0f;
    for(int i=0; i < D; ++i){
        aux += tree[node_idx*(D+1) + i]*points[p*D + i];
        // if(node_idx == 2){
        //     printf("\t%f ",aux);
        // }
    }
    // if(node_idx == 2){
    //     printf("\t>>> %f %f %f | %f %f %f | %f %f\n",tree[node_idx*(D+1)],tree[node_idx*(D+1)+1], tree[node_idx*(D+1)+2], points[p*D], points[p*D + 1], points[p*D + 2], aux, tree[node_idx*(D+1) + D]);
    // }
    
    return aux < tree[node_idx*(D+1) + D];
}


__global__
void build_tree_init(typepoints* tree, unsigned int* points_parent, int* device_child_count, bool* is_leaf, typepoints* points, int* actual_depth, int N, int D)
{
    int tid = blockDim.x*blockIdx.x+threadIdx.x;
    curandState_t r; 
    curand_init(RANDOM_SEED+tid, // the seed controls the sequence of random values that are produced
            blockIdx.x,  // the sequence number is only important with multiple cores 
            tid,  // the offset is how much extra we advance in the sequence for each call, can be 0 
            //   &states[blockIdx.x]);
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
        // printf("Sampled points: %d %d\n", p1,p2);

        create_node(0, p1, p2, tree, points, D);
        *actual_depth = 1;

        is_leaf[0] = false;
        device_child_count[0] = N;
    }

    
}


__global__
void build_tree_update_parents(typepoints* tree,
                               unsigned int* points_parent,
                               bool* is_leaf,
                               int* device_sample_points,
                               int* device_child_count,
                               typepoints* points,
                               int* actual_depth,
                               int N, int D)
{
    int tid = blockDim.x*blockIdx.x+threadIdx.x;
    curandState_t r; 
    curand_init(RANDOM_SEED+tid, // the seed controls the sequence of random values that are produced
            blockIdx.x,  // the sequence number is only important with multiple cores 
            tid,  // the offset is how much extra we advance in the sequence for each call, can be 0 
            //   &states[blockIdx.x]);
            &r);

    int right_child, p;

    // Set nodes parent in the new depth
    for(p = tid; p < N; p+=blockDim.x*gridDim.x){
        // Heap left and right nodes are separeted by 1
        // printf(">%d %d\n", points_parent[p], is_leaf[points_parent[p]]);
        // if(is_leaf[points_parent[p]]){
        if(device_child_count[points_parent[p]] > MAX_TREE_CHILD){
            right_child = check_hyperplane_side(points_parent[p], p, tree, points, D);
            points_parent[p] = HEAP_LEFT(points_parent[p])+right_child;
            atomicAdd(&device_child_count[points_parent[p]],1);
            // device_sample_points[2*points_parent[p] + curand(&r) % 2] = p;
            if(device_sample_points[2*points_parent[p]] == -1){
                device_sample_points[2*points_parent[p]] = N;
            }
            atomicMin(&device_sample_points[2*points_parent[p]], p);
            atomicMax(&device_sample_points[2*points_parent[p] + 1], p);
        }
        __syncwarp();
        // __syncthreads();
    }
}

__global__
void build_tree_post_update_parents(typepoints* tree,
                               unsigned int* points_parent,
                               bool* is_leaf,
                               int* device_sample_points,
                               int* device_child_count,
                               typepoints* points,
                               int* actual_depth,
                               int N, int D)
{
    int tid = blockDim.x*blockIdx.x+threadIdx.x;

    int p;

    
    // Set nodes parent in the new depth
    for(p = tid; p < N; p+=blockDim.x*gridDim.x){
        // device_sample_points[2*points_parent[p]] = p;
        // device_sample_points[2*points_parent[p]+1] = p;

        if(device_child_count[points_parent[p]] <= MAX_TREE_CHILD){
            is_leaf[points_parent[p]] = true;
        }

        // Heap left and right nodes are separeted by 1
        // if(device_sample_points[2*points_parent[p]]     == -1 &&
        //    device_sample_points[2*points_parent[p] + 1] != p){
        //         device_sample_points[2*points_parent[p]] = p;
        // }
        // if(device_sample_points[2*points_parent[p] + 1] == -1 &&
        //    device_sample_points[2*points_parent[p]]     != p){
        //         device_sample_points[2*points_parent[p]+1] = p;
        // }
    }
}

__global__
void build_tree_create_nodes(typepoints* tree,
                             unsigned int* points_parent,
                             bool* is_leaf,
                             int* device_sample_points,
                             int* device_child_count,
                             typepoints* points,
                             int* actual_depth,
                             int N, int D)
{
    int tid = blockDim.x*blockIdx.x+threadIdx.x;
    // curandState_t r; 
    // curand_init(RANDOM_SEED+tid, // the seed controls the sequence of random values that are produced
    //         blockIdx.x,  // the sequence number is only important with multiple cores 
    //         tid,  // the offset is how much extra we advance in the sequence for each call, can be 0 
    //         //   &states[blockIdx.x]);
    //         &r);

    int p1, p2;
    unsigned int bit_mask, i, node_thread;

    // Create new nodes
    for(int nid=tid; nid < pow(2,*actual_depth); nid+=blockDim.x*gridDim.x){
        node_thread = 0; // start on root
        bit_mask = 1;
        // Each thread find the node index to be created
        for(i=1; i <= *actual_depth; ++i){
            node_thread = HEAP_LEFT(node_thread) + ((nid & bit_mask) != 0);
            bit_mask = pow(2,i);
        }

        int parent_id = HEAP_PARENT(node_thread);
        // if(is_leaf[parent_id] && device_child_count[parent_id] > MAX_TREE_CHILD){
        if(device_child_count[node_thread] > MAX_TREE_CHILD){
            p1 = device_sample_points[2*node_thread];
            // p1 = curand(&r) % N;
            // int init_p_search = p1 != -1 ? p1 : 0;
            // while(points_parent[p1] != node_thread){
            //     p1=((p1+1)%N);
            //     if(p1 == init_p_search){
            //         p1 = -1;
            //         break;
            //     }
            // }
            p2 = device_sample_points[2*node_thread + 1];
            // p2 = curand(&r) % N;
            // p2 = p1+1 % N;
            // if(p1 == p2) p2=(p2+1)%N;

            // Ensure that two different points was sampled
            // init_p_search = p2 != -1 ? p2 : 0;
            // while((p1 == p2  || points_parent[p2] != node_thread)){
            //     p2=((p2+1)%N);
            //     if(p2 == init_p_search){
            //         p2 = -1;
            //         break;
            //     }
            // }
            if(p1 != -1 && p2 != -1){
                create_node(node_thread, p1, p2, tree, points, D);
                is_leaf[parent_id] = false;
                // is_leaf[node_thread] = true;
                if(device_child_count[parent_id] > 2500) printf("%d\n", device_child_count[parent_id]);
            }
            else{
                // if(device_child_count[parent_id] > 200)
                printf("%d %d %d | %d %d\n",*actual_depth, parent_id, device_child_count[parent_id], p1,p2);
                device_child_count[parent_id] = 0;
                // printf("BBBBBBBBBBBBB\n");
            }
        }
    }
    
}


__global__
void build_tree_set_all_leafs(typepoints* tree,
                             unsigned int* points_parent,
                             bool* is_leaf,
                             int* device_sample_points,
                             int* device_child_count,
                             typepoints* points,
                             int* actual_depth,
                             int N, int D,
                             int depth)
{
    int tid = blockDim.x*blockIdx.x+threadIdx.x;
    int init_level = pow(2,depth)-1;
    int end_depth_level = pow(2,depth+1)-1;

    for(int node=init_level+tid; node < end_depth_level; node+=blockDim.x*gridDim.x){
        if(device_child_count[node] > 0){
            is_leaf[node] = true;
        }
    }
}

__global__
void
build_tree_utils(int* actual_depth){
    *actual_depth = *actual_depth+1;
}


__global__
void
build_tree_max_leaf_size(int* max_leaf_size, bool* is_leaf, int* device_child_count, int max_nodes)
{
    // __shared__ int s_max_leaf_size;
    int tid = blockDim.x*blockIdx.x+threadIdx.x;

    // if(threadIdx.x == 0){
    //     s_max_leaf_size = 0;
    // }
    // __syncthreads();
    for(int node=tid; node < max_nodes; node+=blockDim.x*gridDim.x){
        if(is_leaf[node]){
            // atomicMax(&s_max_leaf_size, device_child_count[tid]);
            atomicMax(max_leaf_size, device_child_count[node]);
        }
    }
    // __syncthreads();
    // if(threadIdx.x == 0){
    //     atomicMax(max_leaf_size, s_max_leaf_size);
    // }
}

__global__
void build_tree_bucket_points(typepoints* tree,
                               unsigned int* points_parent,
                               bool* is_leaf,
                               int* device_sample_points,
                               int* device_child_count,
                               typepoints* points,
                               int* actual_depth,
                               int* bucket_nodes,
                               int* tmp_node_child_count,
                               int N, int D, int bucket_size)
{
    int tid = blockDim.x*blockIdx.x+threadIdx.x;
    // int right_child, p;
    int my_id_on_bucket, parent_id;
    for(int p = tid; p < N; p+=blockDim.x*gridDim.x){
        parent_id = points_parent[p];
        my_id_on_bucket = atomicAdd(&tmp_node_child_count[parent_id], 1);
        bucket_nodes[parent_id*bucket_size + my_id_on_bucket] = p;
        // printf("%d %d %d | %d\n", parent_id, bucket_size, my_id_on_bucket, parent_id*bucket_size + my_id_on_bucket);
        // if(!is_leaf[points_parent[p]]){
        //     right_child = check_hyperplane_side(points_parent[p], p, tree, points, D);
        //     points_parent[p] = HEAP_LEFT(points_parent[p])+right_child;
        // }
        // atomicAdd(&device_child_count[points_parent[p]],1);
        // device_sample_points[2*points_parent[p] + curand(&r) % 2] = p;
    }
}

__device__
inline
float euclidean_distance_sqr(typepoints* v1, typepoints* v2, int D)
{
    typepoints ret = 0.0f;
    typepoints diff;

    for(int i=0; i < D; ++i){
        diff = v1[i] - v2[i];
        ret += diff;
    }

    return ret;
}

__global__
void compute_knn_from_buckets(typepoints* tree,
                               unsigned int* points_parent,
                               bool* is_leaf,
                               int* device_sample_points,
                               int* device_child_count,
                               typepoints* points,
                               int* actual_depth,
                               int* bucket_nodes,
                               int* knn_indices,
                               typepoints* knn_sqr_dist,
                               int N, int D, int max_bucket_size, int K)
{
    int tid = blockDim.x*blockIdx.x+threadIdx.x;
    int parent_id, bucket_size, max_id_point, tmp_point;
    typepoints max_dist_val, tmp_dist_val;
    // __shared__ int local_knn_indices[];
    // __shared__ typepoints local_knn_sqr_dist[];
    // extern __shared__ int local_knn_indices[];
    // extern __shared__ typepoints local_knn_sqr_dist[];
    int local_knn_indices[MAX_TREE_CHILD];
    typepoints local_knn_sqr_dist[MAX_TREE_CHILD];

    for(int p = tid; p < N; p+=blockDim.x*gridDim.x){
        parent_id = points_parent[p];
        bucket_size = device_child_count[parent_id];
        // __syncthreads();
        for(int i=0; i < K; ++i){
            local_knn_indices[i] = knn_indices[p*K+i];
            local_knn_sqr_dist[i] = knn_sqr_dist[p*K+i];
        }
        // __syncthreads();

        // TODO: Run a first scan?
        max_id_point = 0;
        max_dist_val = local_knn_sqr_dist[0];

        // for(int j=1; j < K; ++j){
        //     if(local_knn_sqr_dist[j] > max_dist_val){
        //         // tmp_dist_val = local_knn_sqr_dist[j];
        //         max_id_point = j;
        //         max_dist_val = local_knn_sqr_dist[j];
        //     }
        // }
        
        for(int i=0; i < bucket_size; ++i){
            tmp_point = bucket_nodes[max_bucket_size*parent_id + i];
            if(p == tmp_point) continue;

            tmp_dist_val = euclidean_distance_sqr(&points[tmp_point], &points[p], D);
            if(tmp_dist_val < max_dist_val){
                local_knn_indices[max_id_point] = tmp_point;
                local_knn_sqr_dist[max_id_point] = tmp_dist_val;

                max_dist_val = tmp_dist_val;
                for(int j=0; j < K; ++j){
                    if(local_knn_sqr_dist[j] > max_dist_val){
                        // tmp_dist_val = local_knn_sqr_dist[j];
                        max_id_point = j;
                        max_dist_val = local_knn_sqr_dist[j];
                    }
                }
            }

            // __syncthreads();
        }
        // __syncthreads();
        for(int i=0; i < K; ++i){
            knn_indices[p*K+i]  = local_knn_indices[i];
            knn_sqr_dist[p*K+i] = local_knn_sqr_dist[i];
        }
        // __syncthreads();
        
    }
}


__global__
void summary_tree(typepoints* tree,
                  bool* is_leaf,
                  int* device_child_count,
                  int* leaf_subtree_count,
                  typepoints* points,
                  int N, int D, int max_nodes)
{
    int tid = blockDim.x*blockIdx.x+threadIdx.x;


    for(int i=tid; i < max_nodes; ++i){
        int node = i;
        while(node>0){
            if(is_leaf[i]){
                atomicAdd(&leaf_subtree_count[node], 1);
            }
            node = HEAP_PARENT(node);
            // __syncthreads();
        }
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

int main(int argc,char* argv[]) {
    // test_random<<<1,1>>>();
    // test_dynamic_vec_reg<<<1,1>>>(15);
    // thrust::device_vector<int> test_var(sizeof(int), 0);
    // test_atomic<<<1,1>>>(thrust::raw_pointer_cast(test_var.data()),15);
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

    std::cout << N << std::endl;
    std::cout << D << std::endl;
    std::cout << std::endl;

    // thrust::copy(knn_indices, knn_indices + num_points * num_neighbors, knn_indices_device.begin());
    // thrust::device_vector<long> knn_indices_long_device(knn_indices_long, knn_indices_long + num_points * num_neighbors);   
    
    std::vector<typepoints> points(N*D);
    std::vector<int> labels(N*D);
    int total_labels = 2;

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

        // printf("%d ",i);
        for(int j=0; j < D; ++j){
            if(type_init == "UNIFORM_SQUARE"){
                points[i*D+j] = rand() % N;
            }
            if(type_init == "SPLITED_LINE"){
                points[i*D+j] = l + (l>N/2)*N/2;
            }
            // printf("%f ", points[i*D+j]);
        }
        // printf("\n");
        labels[i] = (l>N/2);
    }


    // std::vector<typepoints> X_axis(N);
    // std::vector<typepoints> Y_axis(N);
    
    // for(int i=0; i < N; ++i){
    //     X_axis[i] = points[i*D];
    //     Y_axis[i] = points[i*D + 1];
    // }
    // plt::scatter<typepoints,typepoints>(X_axis, Y_axis,10.0);
    // plt::show();

    thrust::device_vector<typepoints> device_points(points.begin(), points.end());
    // test<<<1,100>>>(thrust::raw_pointer_cast(device_points.data()), N, D);
    // thrust::copy(device_points.begin(), device_points.begin() + N*D, points.begin());

    // int MAX_NODES = 2*N-1 - N;


    /* TODO: This is not necessary.
    Since the Tree is not balanced, it is possible to store each level of tree
    in an different array, where new nodes are allocated only if their parents
    are not leafs.
    */
    int MAX_NODES = 0;
    for(int i=0; i < MAX_DEPTH+1; ++i){
        MAX_NODES+=pow(2,i);
    }
    

    std::cout << "MAX NODES: " << MAX_NODES << std::endl;
    // thrust::device_vector<TreeNode> device_tree(sizeof(TreeNode) * (MAX_NODES));
    thrust::device_vector<typepoints> device_tree((D + 1) * (MAX_NODES));
    thrust::device_vector<bool> device_is_leaf(MAX_NODES, false);
    thrust::device_vector<int> device_sample_points(MAX_NODES*2);
    thrust::device_vector<int> device_child_count(MAX_NODES, 0);
    thrust::device_vector<int> device_actual_depth(1,0);
    thrust::device_vector<unsigned int> device_points_parent(N, 0);

    const int nt = 1024;
    const int mp = 8;

    // build_tree<<<mp,nt>>>(thrust::raw_pointer_cast(device_tree.data()),
    //                      thrust::raw_pointer_cast(device_points_parent.data()),
    //                      thrust::raw_pointer_cast(device_is_leaf.data()),
    //                      thrust::raw_pointer_cast(device_points.data()),
    //                      thrust::raw_pointer_cast(device_actual_depth.data()),
    //                      N, D);
    // CudaTest("Build Tree Kernel failed!");

    build_tree_init<<<mp,nt>>>(thrust::raw_pointer_cast(device_tree.data()),
                               thrust::raw_pointer_cast(device_points_parent.data()),
                               thrust::raw_pointer_cast(device_child_count.data()),
                               thrust::raw_pointer_cast(device_is_leaf.data()),
                               thrust::raw_pointer_cast(device_points.data()),
                               thrust::raw_pointer_cast(device_actual_depth.data()),
                               N, D);
    cudaDeviceSynchronize();
    CudaTest((char *)"build_tree_init Kernel failed!");

    if(VERBOSE >= 1){
        std::cout << std::endl;
    }

    Cron total_cron;
    Cron update_parents_cron;
    Cron create_nodes_cron;

    total_cron.start();
    int depth;
    for(depth=1; depth < MAX_DEPTH; ++depth){
        thrust::fill(device_sample_points.begin(), device_sample_points.end(), -1);
        // thrust::fill(device_child_count.begin(), device_child_count.end(), 0);
        cudaDeviceSynchronize();
        update_parents_cron.start();
        build_tree_update_parents<<<mp,nt>>>(thrust::raw_pointer_cast(device_tree.data()),
                                   thrust::raw_pointer_cast(device_points_parent.data()),
                                   thrust::raw_pointer_cast(device_is_leaf.data()),
                                   thrust::raw_pointer_cast(device_sample_points.data()),
                                   thrust::raw_pointer_cast(device_child_count.data()),
                                   thrust::raw_pointer_cast(device_points.data()),
                                   thrust::raw_pointer_cast(device_actual_depth.data()),
                                   N, D);
        cudaDeviceSynchronize();
        CudaTest((char *)"build_tree_update_parents Kernel failed!");
        build_tree_post_update_parents<<<mp,nt>>>(thrust::raw_pointer_cast(device_tree.data()),
                                   thrust::raw_pointer_cast(device_points_parent.data()),
                                   thrust::raw_pointer_cast(device_is_leaf.data()),
                                   thrust::raw_pointer_cast(device_sample_points.data()),
                                   thrust::raw_pointer_cast(device_child_count.data()),
                                   thrust::raw_pointer_cast(device_points.data()),
                                   thrust::raw_pointer_cast(device_actual_depth.data()),
                                   N, D);
        cudaDeviceSynchronize();
        update_parents_cron.stop();
        CudaTest((char *)"build_tree_post_update_parents Kernel failed!");
        
        create_nodes_cron.start();
        build_tree_create_nodes<<<mp,nt>>>(thrust::raw_pointer_cast(device_tree.data()),
                                   thrust::raw_pointer_cast(device_points_parent.data()),
                                   thrust::raw_pointer_cast(device_is_leaf.data()),
                                   thrust::raw_pointer_cast(device_sample_points.data()),
                                   thrust::raw_pointer_cast(device_child_count.data()),
                                   thrust::raw_pointer_cast(device_points.data()),
                                   thrust::raw_pointer_cast(device_actual_depth.data()),
                                   N, D);
        cudaDeviceSynchronize();
        create_nodes_cron.stop();
        CudaTest((char *)"build_tree_create_nodes Kernel failed!");

        build_tree_utils<<<1,1>>>(thrust::raw_pointer_cast(device_actual_depth.data()));
        cudaDeviceSynchronize();
        CudaTest((char *)"build_tree_utils Kernel failed!");

        if(VERBOSE >= 1){
            std::cout << "\e[ABuilding Tree Depth: " << depth+1 << "/" << MAX_DEPTH << std::endl;
        }
    }

    // thrust::fill(device_child_count.begin(), device_child_count.end(), 0);

    cudaDeviceSynchronize();
    update_parents_cron.start();
    build_tree_update_parents<<<mp,nt>>>(thrust::raw_pointer_cast(device_tree.data()),
                                thrust::raw_pointer_cast(device_points_parent.data()),
                                thrust::raw_pointer_cast(device_is_leaf.data()),
                                thrust::raw_pointer_cast(device_sample_points.data()),
                                thrust::raw_pointer_cast(device_child_count.data()),
                                thrust::raw_pointer_cast(device_points.data()),
                                thrust::raw_pointer_cast(device_actual_depth.data()),
                                N, D);
    cudaDeviceSynchronize();
    CudaTest((char *)"build_tree_update_parents Kernel failed!");
    build_tree_post_update_parents<<<mp,nt>>>(thrust::raw_pointer_cast(device_tree.data()),
                                thrust::raw_pointer_cast(device_points_parent.data()),
                                thrust::raw_pointer_cast(device_is_leaf.data()),
                                thrust::raw_pointer_cast(device_sample_points.data()),
                                thrust::raw_pointer_cast(device_child_count.data()),
                                thrust::raw_pointer_cast(device_points.data()),
                                thrust::raw_pointer_cast(device_actual_depth.data()),
                                N, D);
    cudaDeviceSynchronize();
    update_parents_cron.stop();
    CudaTest((char *)"build_tree_update_parents Kernel failed!");
    total_cron.stop();



    cudaDeviceSynchronize();
    



    // Check what leaf nodes in tree have childrens and set device_is_leaf when a node have more than 0 children
    // create_nodes_cron.start();
    // build_tree_set_all_leafs<<<mp,nt>>>(thrust::raw_pointer_cast(device_tree.data()),
    //                             thrust::raw_pointer_cast(device_points_parent.data()),
    //                             thrust::raw_pointer_cast(device_is_leaf.data()),
    //                             thrust::raw_pointer_cast(device_sample_points.data()),
    //                             thrust::raw_pointer_cast(device_child_count.data()),
    //                             thrust::raw_pointer_cast(device_points.data()),
    //                             thrust::raw_pointer_cast(device_actual_depth.data()),
    //                             N, D, depth-1);
    // cudaDeviceSynchronize();
    // CudaTest((char *)"build_tree_set_all_leafs Kernel failed!");


    
    Cron cron_classify_points;
    cron_classify_points.start();

    thrust::device_vector<int> device_max_leaf_size(1,0);
    
    // int max_child = *(thrust::max_element(device_child_count.begin()+pow(2,depth-3)-1, device_child_count.end()));
    build_tree_max_leaf_size<<<mp,nt>>>(thrust::raw_pointer_cast(device_max_leaf_size.data()),
                                thrust::raw_pointer_cast(device_is_leaf.data()),
                                thrust::raw_pointer_cast(device_child_count.data()),
                                MAX_NODES);
    cudaDeviceSynchronize();
    CudaTest((char *)"build_tree_max_leaf_size Kernel failed!");
    int max_child;
    thrust::copy(device_max_leaf_size.begin(), device_max_leaf_size.begin()+1, &max_child);

    std::cout << "Max child count: " << max_child << "\n" << std::endl;



    int total_leafs = thrust::reduce(device_is_leaf.begin(), device_is_leaf.end(), 0.0, thrust::plus<float>());
    cudaDeviceSynchronize();

    thrust::device_vector<int> device_leaf_subtree_count(MAX_NODES, 0);
    thrust::device_vector<int> device_compact_buckets(total_leafs*max_child, 0);
    thrust::device_vector<int> device_compact_buckets_idx(total_leafs, 0);
    Cron cron_summary_tree_leafsubtreecount;
    cron_summary_tree_leafsubtreecount.start();


    std::cout << "Total leafs: " << total_leafs << std::endl;


    summary_tree<<<mp,nt>>>(thrust::raw_pointer_cast(device_tree.data()),
                            thrust::raw_pointer_cast(device_is_leaf.data()),
                            thrust::raw_pointer_cast(device_child_count.data()),
                            thrust::raw_pointer_cast(device_leaf_subtree_count.data()),
                            thrust::raw_pointer_cast(device_points.data()),
                            N, D, MAX_NODES);
    cudaDeviceSynchronize();
    cron_summary_tree_leafsubtreecount.stop();
    CudaTest((char *)"summary_tree Kernel failed!");




    
    thrust::device_vector<int> device_nodes_buckets(MAX_NODES*max_child, -1);
    thrust::device_vector<int> device_count_buckets(MAX_NODES, 0);
    build_tree_bucket_points<<<mp,nt>>>(thrust::raw_pointer_cast(device_tree.data()),
                                thrust::raw_pointer_cast(device_points_parent.data()),
                                thrust::raw_pointer_cast(device_is_leaf.data()),
                                thrust::raw_pointer_cast(device_sample_points.data()),
                                thrust::raw_pointer_cast(device_child_count.data()),
                                thrust::raw_pointer_cast(device_points.data()),
                                thrust::raw_pointer_cast(device_actual_depth.data()),
                                thrust::raw_pointer_cast(device_nodes_buckets.data()),
                                thrust::raw_pointer_cast(device_count_buckets.data()),
                                N, D, max_child);
    cudaDeviceSynchronize();
    cron_classify_points.stop();
    CudaTest((char *)"build_tree_bucket_points Kernel failed!");
    std::vector<int> nodes_buckets(MAX_NODES*max_child);
    thrust::copy(device_nodes_buckets.begin(), device_nodes_buckets.begin() + MAX_NODES*max_child, nodes_buckets.begin());
    std::vector<int> count_buckets(MAX_NODES);
    thrust::copy(device_count_buckets.begin(), device_count_buckets.begin() + MAX_NODES, count_buckets.begin());
    cudaDeviceSynchronize();

    // device_nodes_buckets.clear();
    // device_nodes_buckets.shrink_to_fit();
    // device_tree.clear();
    // device_tree.shrink_to_fit();

    // create_nodes_cron.stop();
    int K = 8;

    // int a;std::cin >> a;
    
    
    thrust::device_vector<int> device_knn_indices(N*K, -1);
    std::vector<int> knn_indices(N*K);
    thrust::device_vector<typepoints> device_knn_sqr_distances(N*K, FLT_MAX); // 0x7f800000 is a "infinite" float value
    std::vector<int> knn_sqr_distances(N*K);
    // std::cin >> a;

    Cron cron_knn;
    cron_knn.start();

    compute_knn_from_buckets<<<mp,nt>>>(thrust::raw_pointer_cast(device_tree.data()),
                                thrust::raw_pointer_cast(device_points_parent.data()),
                                thrust::raw_pointer_cast(device_is_leaf.data()),
                                thrust::raw_pointer_cast(device_sample_points.data()),
                                thrust::raw_pointer_cast(device_child_count.data()),
                                thrust::raw_pointer_cast(device_points.data()),
                                thrust::raw_pointer_cast(device_actual_depth.data()),
                                thrust::raw_pointer_cast(device_nodes_buckets.data()),
                                thrust::raw_pointer_cast(device_knn_indices.data()),
                                thrust::raw_pointer_cast(device_knn_sqr_distances.data()),
                                N, D, max_child, K);
    cudaDeviceSynchronize();
    cron_knn.stop();    
    CudaTest((char *)"compute_knn_from_buckets Kernel failed!");

    thrust::copy(device_knn_indices.begin(), device_knn_indices.begin() + N*K, knn_indices.begin());
    thrust::copy(knn_sqr_distances.begin(), knn_sqr_distances.begin() + N*K, knn_sqr_distances.begin());
    cudaDeviceSynchronize();

    std::cout << "Points neighbors" << std::endl;
    for(int i=0; i < 4*8; i+=4){
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
    




    if(VERBOSE >= 1){
        printf("Build Tree Kernel takes %lf seconds\n", create_nodes_cron.t_total/1000);
        printf("Update parents Kernel takes %lf seconds\n", update_parents_cron.t_total/1000);
        printf("Create nodes Kernel takes %lf seconds\n", create_nodes_cron.t_total/1000);
        printf("Bucket creation Kernel takes %lf seconds\n", cron_classify_points.t_total/1000);
        printf("Summary Leaf subtree count computation Kernel takes %lf seconds\n", cron_summary_tree_leafsubtreecount.t_total/1000);
        printf("KNN computation Kernel takes %lf seconds\n", cron_knn.t_total/1000);
        thrust::copy(device_points_parent.begin(), device_points_parent.begin() + N, labels.begin());
        total_labels = countDistinct(labels.data(),N);
        std::cout << "Total clusters: " << total_labels << std::endl;
    }

    if(VERBOSE >= 2){

        set<int> s; 
    
        // for (int i = 0; i < N; i++) { 
        //     s.insert(labels[i]); 
        // }
        // set<int>::iterator it; 
        
        // for (it = s.begin(); it != s.end(); ++it){
        for(int b=0; b < MAX_NODES; ++b){
            // int l = (int) *it; 
            int count_cluster;
            // for(int i=0; i < N; ++i){
                // if(labels[i] == l) count_cluster++;
            // }
            count_cluster = count_buckets[b];
            if(count_cluster == 0) continue;

            std::vector<typepoints> X_axis(count_cluster);
            std::vector<typepoints> Y_axis(count_cluster);
            // std::cerr << "AAAA " << count_cluster << std::endl;
            
            // int j =0;
            // for(int i=0; i < N; ++i){
            //     if(labels[i] == l){
            //         X_axis[j] = points[i*D];
            //         Y_axis[j] = points[i*D + 1];
            //         ++j;
            //     }
            // }

            for(int i=0; i < count_cluster; ++i){
                int p = nodes_buckets[b*max_child+i];
                // std::cerr << "BBBB " << p << " "<< count_cluster << std::endl;
                X_axis[i] = points[D*p];
                Y_axis[i] = points[D*p + 1];
            }
            plt::scatter<typepoints,typepoints>(X_axis, Y_axis,1.0,{
                {"alpha", "0.8"}
            });

        }
        // plt::show();
        plt::save("out.png");
    }
}
