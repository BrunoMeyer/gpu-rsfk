#ifndef __BUILD_TREE_INIT__CU
#define __BUILD_TREE_INIT__CU

#include "create_node.cu"

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
                     int* depth_level_count,
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
        depth_level_count[0] = 1;
        // depth_level_count[1] = 2;

        is_leaf[0] = false;
        child_count[0] = N;
    }
}

#endif