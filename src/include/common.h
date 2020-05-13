#ifndef __COMMON_RPFK__H
#define __COMMON_RPFK__H

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


// EDV = EUCLIDIEAN DISTANCE VERSION
#define EDV_ATOMIC_OK               0
#define EDV_ATOMIC_CSE              1   // common subexpression elimination
#define EDV_NOATOMIC                2
#define EDV_NOATOMIC_NOSHM          3   // value returned in register (NO SHM)
#define EDV_WARP_REDUCE_XOR         4
#define EDV_WARP_REDUCE_XOR_NOSHM   5   // value returned in register (NO SHM)

#define EUCLIDEAN_DISTANCE_VERSION EDV_WARP_REDUCE_XOR_NOSHM


#define RELEASE 0
#define DEBUG 1
#define COMPILE_TYPE RELEASE

#endif