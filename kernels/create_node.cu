#ifndef __CREATE_NODE__CU
#define __CREATE_NODE__CU

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
    *tree_count = 0;

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

#endif