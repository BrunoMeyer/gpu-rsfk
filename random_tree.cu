#ifndef __RANDOM_TREE__CU
#define __RANDOM_TREE__CU

#include <curand.h>
#include <curand_kernel.h>

#include "kernels/create_node.cu"

__global__
void test_random(int RANDOM_SEED)
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
    register int arr[1024];
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
void test(typepoints* arr, int N, int D, int RANDOM_SEED)
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

__global__
void test_cuda_dynamic_declaration(int* arr, int N)
{
    for(int i=0; i < N; ++i){
        arr[i] = i*10+i;
    }
}


















// __global__
// void build_tree_set_all_leafs(typepoints* tree,
//                               int* tree_parents,
//                               int* tree_children,
//                               int* points_parent,
//                               int* is_right_child,
//                               bool* is_leaf,
//                               int* sample_points,
//                               int* child_count,
//                               typepoints* points,
//                               int* actual_depth,
//                               int* tree_count,
//                               int* depth_level_count,
//                               int* count_new_nodes,
//                               int N, int D)
// {
//     int tid = blockDim.x*blockIdx.x+threadIdx.x;
//     int init_level = pow(2,depth)-1;
//     int end_depth_level = pow(2,depth+1)-1;

//     for(int node=init_level+tid; node < end_depth_level; node+=blockDim.x*gridDim.x){
//         if(device_child_count[node] > 0){
//             is_leaf[node] = true;
//         }
//     }
// }


















// __global__
// void summary_tree(typepoints* tree,
//                   bool* is_leaf,
//                   int* device_child_count,
//                   int* leaf_subtree_count,
//                   typepoints* points,
//                   int N, int D, int max_nodes)
// {
//     int tid = blockDim.x*blockIdx.x+threadIdx.x;


//     for(int i=tid; i < max_nodes; ++i){
//         int node = i;
//         while(node>0){
//             if(is_leaf[i]){
//                 atomicAdd(&leaf_subtree_count[node], 1);
//             }
//             node = HEAP_PARENT(node);
//             // __syncthreads();
//         }
//     }
// }








/*
__global__
void build_tree_create_nodes(typepoints* tree,
                             int* tree_parents,
                             int* tree_children,
                             int* points_parent,
                             bool* is_leaf,
                             int* device_sample_points,
                             int* device_child_count,
                             typepoints* points,
                             int* actual_depth,
                             int* tree_count,
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
    int bit_mask, i, node_thread;

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
                // create_node(node_thread, p1, p2, tree, points, D);
                create_node(parent, is_right_child, tree, tree_parents,
                            tree_children, tree_count, p1, p2, points, D);

                 

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
*/

#endif