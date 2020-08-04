#ifndef __COMMON_RSFK__H
#define __COMMON_RSFK__H

#define RSFK_typepoints float

// Lines per column
#define RSFK_N_D 0

// Columns 
#define RSFK_D_N 1

#define RSFK_POINTS_STRUCTURE N_D
#define RSFK_TREE_STRUCTURE N_D

#if   RSFK_POINTS_STRUCTURE == D_N
    #define get_point_idx(point,dimension,N,D) (dimension*N+point)
#elif RSFK_POINTS_STRUCTURE == N_D
    #define get_point_idx(point,dimension,N,D) (point*D+dimension)
#endif

#if   RSFK_TREE_STRUCTURE == D_N
    #define get_tree_idx(nidx,dimension,N,D) (dimension*N+nidx)
#elif RSFK_TREE_STRUCTURE == N_D
    #define get_tree_idx(nidx,dimension,N,D) (nidx*(D+1)+dimension)
#endif


// EDV = EUCLIDIEAN DISTANCE VERSION
#define RSFK_EDV_ATOMIC_OK               0
#define RSFK_EDV_ATOMIC_CSE              1   // common subexpression elimination
#define RSFK_EDV_NOATOMIC                2
#define RSFK_EDV_NOATOMIC_NOSHM          3   // value returned in register (NO SHM)
#define RSFK_EDV_WARP_REDUCE_XOR         4
#define RSFK_EDV_WARP_REDUCE_XOR_NOSHM   5   // value returned in register (NO SHM)

#define RSFK_EUCLIDEAN_DISTANCE_VERSION RSFK_EDV_WARP_REDUCE_XOR_NOSHM


#define RSFK_RELEASE 0
#define RSFK_DEBUG 1
#define RSFK_COMPILE_TYPE RELEASE

#endif