#ifndef __CREATE_NODE__CU
#define __CREATE_NODE__CU

#include "../include/common.h"

__device__
inline
void create_root(RSFK_typepoints* tree,
                 int* tree_parents,
                 int* tree_children,
                 int* tree_count,
                 int p1,
                 int p2,
                 int* count_new_nodes,
                 RSFK_typepoints* points,
                 int D, int N)
{
    // Average point
    // node_path*D*2 : D*2 = size of centroid point and normal vector

    int node_idx = 0;
    
    tree_parents[node_idx] = -1;
    *tree_count = 0;

    int i;
    // tree[node_idx*(D+1) + D] = 0.0f;
    tree[get_tree_idx(node_idx,D,*count_new_nodes,D)] = 0.0f;

    for(i=0;i < D; ++i){
        tree[get_tree_idx(node_idx,i,*count_new_nodes,D)] = points[get_point_idx(p1,i,N,D)]-points[get_point_idx(p2,i,N,D)];
        tree[get_tree_idx(node_idx,D,*count_new_nodes,D)]+= tree[get_tree_idx(node_idx,i,*count_new_nodes,D)]*(points[get_point_idx(p1,i,N,D)]+points[get_point_idx(p2,i,N,D)])/2; // multiply the point of plane and the normal vector 
    }
}

__device__
inline
void create_node(int parent,
                 int is_right_child,
                 RSFK_typepoints* tree,
                 int* tree_parents,
                 int* tree_children,
                 int* tree_count,
                 int* count_new_nodes,
                 int p1,
                 int p2,
                 RSFK_typepoints* points,
                 int D, int N)
{
    // Average point
    // node_path*D*2 : D*2 = size of centroid point and normal vector

    int node_idx = atomicAdd(tree_count, 1);
    tree_parents[node_idx] = parent;
    
    tree_children[2*parent+is_right_child] = node_idx;
    int i;

    RSFK_typepoints s = 0.0f;
    for(i=0; i < D; ++i){
        tree[get_tree_idx(node_idx,i,*count_new_nodes,D)] = points[get_point_idx(p1,i,N,D)]-points[get_point_idx(p2,i,N,D)];
        s+= tree[get_tree_idx(node_idx,i,*count_new_nodes,D)]*(points[get_point_idx(p1,i,N,D)]+points[get_point_idx(p2,i,N,D)])/2; // multiply the point of plane and the normal vector 
    }
    tree[get_tree_idx(node_idx,D,*count_new_nodes,D)] = s;
}

#endif