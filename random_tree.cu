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


#include <curand.h>
#include <curand_kernel.h>

#include <stdio.h>
// #include <bits/stdc++.h> 


// #define typepoints float
// // #define RANDOM_SEED 42
// #define RANDOM_SEED 0
// #define MAX_K 1024
// #define MAX_TREE_CHILD 64



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
void create_root(typepoints* tree,
                 int* tree_parents,
                 int* tree_children,
                 int* tree_count,
                 int p1,
                 int p2,
                 typepoints* points,
                 int D)
{
    // Average point
    // node_path*D*2 : D*2 = size of centroid point and normal vector

    int node_idx = 0;
    
    tree_parents[node_idx] = -1;
    *tree_count = 1;

    int i;
    tree[node_idx*(D+1) + D] = 0.0f;

    for(i=0;i < D; ++i){
        tree[node_idx*(D+1) + i] = points[p1*D+i]-points[p2*D+i];
        tree[node_idx*(D+1) + D]+= tree[node_idx*(D+1) + i]*(points[p1*D+i]+points[p2*D+i])/2; // multiply the point of plane and the normal vector 
    }
}

__device__
inline
void create_node(int parent,
                 int is_right_child,
                 typepoints* tree,
                 int* tree_parents,
                 int* tree_children,
                 int* tree_count,
                 int* count_new_nodes,
                 int p1,
                 int p2,
                 typepoints* points,
                 int D)
{
    // Average point
    // node_path*D*2 : D*2 = size of centroid point and normal vector

    int node_idx = atomicAdd(tree_count, 1);
    // atomicAdd(count_new_nodes, 1);
    tree_parents[node_idx] = parent;
    
    tree_children[2*parent+is_right_child] = node_idx;
    // printf("DEBUG1: %d %d\n",2*parent+is_right_child, node_idx);
    int i;
    // typepoints mean_axis_val;
    // typepoints normal_vector_axis_val;
    // typepoints plane_bias = 0;
    tree[node_idx*(D+1) + D] = 0.0f;

    for(i=0; i < D; ++i){
        // mean_axis_val = (points[p1*D+i]+points[p2*D+i])/2;
        // normal_vector_axis_val = points[p2*D+i]-points[p1*D+i];
        // tree[node_idx*(D+1) + i] = normal_vector_axis_val;
        // plane_bias+= tree[node_idx*(D+1) + i]*(points[p1*D+i]+points[p2*D+i])/2; // multiply the point of plane and the normal vector 
        tree[node_idx*(D+1) + i] = points[p1*D+i]-points[p2*D+i];
        tree[node_idx*(D+1) + D]+= tree[node_idx*(D+1) + i]*(points[p1*D+i]+points[p2*D+i])/2; // multiply the point of plane and the normal vector 
    }
}


__device__
inline
int check_hyperplane_side(int node_idx, int p, typepoints* tree, typepoints* points, int D)
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
void build_tree_init(typepoints* tree,
                     int* tree_parents,
                     int* tree_children,
                     int* points_parent,
                     int* child_count,
                     bool* is_leaf,
                     typepoints* points,
                     int* actual_depth,
                     int* tree_count,
                     int* pointer_depth_level,
                     int N, int D)
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

        // create_node(-1, tree,tree_parents, tree_children, tree_count, p1, p2, points, D);
        create_root(tree, tree_parents, tree_children, tree_count, p1, p2, points, D);
        *actual_depth = 1;
        pointer_depth_level[0] = 0;
        pointer_depth_level[1] = 1;

        is_leaf[0] = false;
        child_count[0] = N;

    }
}

__global__
void build_tree_check_points_side(typepoints* tree,
                                  int* tree_parents,
                                  int* tree_children,
                                  int* points_parent,
                                  int* is_right_child,
                                  bool* is_leaf,
                                  int* sample_points,
                                  int* child_count,
                                  typepoints* points,
                                  int* actual_depth,
                                  int* tree_count,
                                  int* pointer_depth_level,
                                  int N, int D)
{
    int tid = blockDim.x*blockIdx.x+threadIdx.x;
    // curandState_t r; 
    // curand_init(RANDOM_SEED+tid, // the seed controls the sequence of random values that are produced
    //         blockIdx.x,  // the sequence number is only important with multiple cores 
    //         tid,  // the offset is how much extra we advance in the sequence for each call, can be 0 
    //         //   &states[blockIdx.x]);
    //         &r);

    int p, is_right;

    // Set nodes parent in the new depth
    for(p = tid; p < N; p+=blockDim.x*gridDim.x){
        is_right = check_hyperplane_side(points_parent[p], p, tree, points, D);
        is_right_child[p] = is_right;
        // atomicAdd(&child_count[2*points_parent[p]+is_right],1);

        // Threats to Validity: This assumes that all the follow properties are false:
        // - The atomic operations assumes an arbitrary/random order
        // - The points are shuffled

        if(sample_points[4*points_parent[p]  + 2*is_right    ] == -1){
            sample_points[4*points_parent[p]  + 2*is_right    ] = N;
        }

        // printf("DEBUG2: %d\n",points_parent[p]);
        atomicMin(&sample_points[4*points_parent[p]  + 2*is_right    ], p);
        atomicMax(&sample_points[4*points_parent[p]  + 2*is_right + 1], p);

        // if(device_child_count[2*points_parent[p]+] > MAX_TREE_CHILD){
        //     right_child = check_hyperplane_side(points_parent[p], p, tree, points, D);
        //     // points_parent[p] = HEAP_LEFT(points_parent[p])+right_child;
        //     points_parent[p] = tree_children[2*points_parent[p]+right_child];

        //     atomicAdd(&device_child_count[points_parent[p]],1);
        //     // device_sample_points[2*points_parent[p] + curand(&r) % 2] = p;
        //     if(device_sample_points[2*points_parent[p]] == -1){
        //         device_sample_points[2*points_parent[p]] = N;
        //     }
        //     atomicMin(&device_sample_points[2*points_parent[p]], p);
        //     atomicMax(&device_sample_points[2*points_parent[p] + 1], p);
        // }

        // __syncwarp();
        // __syncthreads();
    }
}

__global__
void build_tree_count_new_nodes(typepoints* tree,
                                int* tree_parents,
                                int* tree_children,
                                int* points_parent,
                                int* is_right_child,
                                bool* is_leaf,
                                int* sample_points,
                                int* child_count,
                                typepoints* points,
                                int* actual_depth,
                                int* tree_count,
                                int* pointer_depth_level,
                                int* count_new_nodes,
                                int N, int D)
{
    int tid = blockDim.x*blockIdx.x+threadIdx.x;
    for(int node_thread = tid+pointer_depth_level[*actual_depth - 1]; node_thread < pointer_depth_level[*actual_depth]; node_thread+=blockDim.x*gridDim.x){
        if(child_count[node_thread] > MAX_TREE_CHILD){
            if((sample_points[4*node_thread+0] != -1 && sample_points[4*node_thread+1] != -1)) atomicAdd(count_new_nodes, 1);
            if((sample_points[4*node_thread+2] != -1 && sample_points[4*node_thread+3] != -1)) atomicAdd(count_new_nodes, 1);
        }
    }
}

__global__
void build_tree_create_nodes(typepoints* tree,
                             int* tree_parents,
                             int* tree_children,
                             int* points_parent,
                             int* is_right_child,
                             bool* is_leaf,
                             int* sample_points,
                             int* child_count,
                             typepoints* points,
                             int* actual_depth,
                             int* tree_count,
                             int* pointer_depth_level,
                             int* count_new_nodes,
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
    int i, node_thread;

    // if(tid == 0){
    //     printf("DEBUG 7: %d %d %d %d\n", pointer_depth_level[*actual_depth-1], pointer_depth_level[*actual_depth], *count_new_nodes, *actual_depth);
    // }
    // Create new nodes
    for(node_thread = tid+pointer_depth_level[*actual_depth-1]; node_thread < pointer_depth_level[*actual_depth]; node_thread+=blockDim.x*gridDim.x){

        int parent_id = tree_parents[node_thread];
        // if(is_leaf[parent_id] && device_child_count[parent_id] > MAX_TREE_CHILD){
        
        // printf("DEBUG3: %d %d\n", node_thread, child_count[node_thread]);
        if(child_count[node_thread] > MAX_TREE_CHILD){
            for(int is_right=0; is_right < 2; ++is_right){
                p1 = sample_points[4*node_thread + 2*is_right];
                // p1 = curand(&r) % N;
                // int init_p_search = p1 != -1 ? p1 : 0;
                // while(points_parent[p1] != node_thread){
                //     p1=((p1+1)%N);
                //     if(p1 == init_p_search){
                //         p1 = -1;
                //         break;
                //     }
                // }
                p2 = sample_points[4*node_thread + 2*is_right + 1];
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
                    create_node(node_thread, is_right, tree, tree_parents,
                                tree_children, tree_count, count_new_nodes, p1, p2, points, D);

                    is_leaf[node_thread] = false;
                    is_leaf[tree_children[2*node_thread+is_right]] = true;

                    // is_leaf[node_thread] = true;
                    // if(child_count[parent_id] > 2500) printf("%d\n", child_count[parent_id]);
                }
                else{
                    // if(device_child_count[parent_id] > 200)
                    printf("%d %d %d | %d %d\n",*actual_depth, node_thread, child_count[node_thread], p1,p2);
                    child_count[node_thread] = 0;
                    // printf("BBBBBBBBBBBBB\n");
                }
            }
        }
    }
    
}

__global__
void build_tree_update_parents(typepoints* tree,
                               int* tree_parents,
                               int* tree_children,
                               int* points_parent,
                               int* is_right_child,
                               bool* is_leaf,
                               int* sample_points,
                               int* child_count,
                               typepoints* points,
                               int* actual_depth,
                               int* tree_count,
                               int* pointer_depth_level,
                               int* count_new_nodes,
                               int N, int D)
{
    int tid = blockDim.x*blockIdx.x+threadIdx.x;
    // curandState_t r; 
    // curand_init(RANDOM_SEED+tid, // the seed controls the sequence of random values that are produced
    //         blockIdx.x,  // the sequence number is only important with multiple cores 
    //         tid,  // the offset is how much extra we advance in the sequence for each call, can be 0 
    //         //   &states[blockIdx.x]);
    //         &r);

    int right_child, p;

    // Set nodes parent in the new depth
    for(p = tid; p < N; p+=blockDim.x*gridDim.x){
        // Heap left and right nodes are separeted by 1
        // printf(">%d %d\n", points_parent[p], is_leaf[points_parent[p]]);
        // if(is_leaf[points_parent[p]]){

        // TODO: Verify if the "if" statement is necessary
        if(child_count[points_parent[p]] > MAX_TREE_CHILD){
            right_child = is_right_child[p];
            // printf("DEBUG4: %d %d %d %d\n", points_parent[p], 2*points_parent[p]+right_child, tree_children[2*points_parent[p]+right_child], child_count[points_parent[p]]);
            // points_parent[p] = HEAP_LEFT(points_parent[p])+right_child;
            points_parent[p] = tree_children[2*points_parent[p]+right_child];
            // printf("%d %d %d\n", p, is_right_child[p], points_parent[p]);
            atomicAdd(&child_count[points_parent[p]],1);
        }
        __syncwarp();
        // __syncthreads();
    }
}

__global__
void build_tree_post_update_parents(typepoints* tree,
                                    int* tree_parents,
                                    int* tree_children,
                                    int* points_parent,
                                    int* is_right_child,
                                    bool* is_leaf,
                                    int* sample_points,
                                    int* child_count,
                                    typepoints* points,
                                    int* actual_depth,
                                    int* tree_count,
                                    int* pointer_depth_level,
                                    int* count_new_nodes,
                                    int N, int D)
{
    int tid = blockDim.x*blockIdx.x+threadIdx.x;

    int p;

    
    // Set nodes parent in the new depth
    for(p = tid; p < N; p+=blockDim.x*gridDim.x){
        // device_sample_points[2*points_parent[p]] = p;
        // device_sample_points[2*points_parent[p]+1] = p;

        if(child_count[points_parent[p]] <= MAX_TREE_CHILD){
            is_leaf[points_parent[p]] = true;
            // printf("DEBUG5: %d\n", points_parent[p]);
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
//                               int* pointer_depth_level,
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

__global__
void
build_tree_utils(int* actual_depth, int* pointer_depth_level, int* count_new_nodes){
    *actual_depth = *actual_depth+1;
    pointer_depth_level[*actual_depth] = pointer_depth_level[*actual_depth - 1] + *count_new_nodes;
    *count_new_nodes = 0;
}


__global__
void
build_tree_max_leaf_size(int* max_leaf_size, bool* is_leaf, int* child_count, int* count_nodes)
{
    // __shared__ int s_max_leaf_size;
    int tid = blockDim.x*blockIdx.x+threadIdx.x;

    // if(threadIdx.x == 0){
    //     s_max_leaf_size = 0;
    // }
    // __syncthreads();
    for(int node=tid; node < *count_nodes; node+=blockDim.x*gridDim.x){
        if(is_leaf[node]){
            // atomicMax(&s_max_leaf_size, device_child_count[tid]);
            atomicMax(max_leaf_size, child_count[node]);
            // printf("DEBUG6: %d\n", child_count[node]);
        }
    }
    // __syncthreads();
    // if(threadIdx.x == 0){
    //     atomicMax(max_leaf_size, s_max_leaf_size);
    // }
}

__global__
void build_tree_bucket_points(typepoints* tree,
                               int* points_parent,
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
                               int* points_parent,
                               bool* is_leaf,
                               int* sample_points,
                               int* child_count,
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
        bucket_size = child_count[parent_id];
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