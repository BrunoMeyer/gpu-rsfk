/*
This file is part of the GPU-RSFK Project (https://github.com/BrunoMeyer/gpu-rsfk).

BSD 3-Clause License

Copyright (c) 2021, Bruno Henrique Meyer, Wagner M. Nunan Zola
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef __COMMON_RSFK__H
#define __COMMON_RSFK__H

#define RSFK_typepoints float

// Lines per column
#define RSFK_N_D 0

// Columns 
#define RSFK_D_N 1

#define RSFK_POINTS_STRUCTURE RSFK_N_D
#define RSFK_TREE_STRUCTURE RSFK_N_D

#if   RSFK_POINTS_STRUCTURE == RSFK_D_N
    #define get_point_idx(point,dimension,N,D) (dimension*N+point)
#elif RSFK_POINTS_STRUCTURE == RSFK_N_D
    #define get_point_idx(point,dimension,N,D) (point*D+dimension)
#endif

#if   RSFK_TREE_STRUCTURE == RSFK_D_N
    #define get_tree_idx(nidx,dimension,N,D) (dimension*N+nidx)
#elif RSFK_TREE_STRUCTURE == RSFK_N_D
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
#define RSFK_COMPILE_TYPE RSFK_RELEASE

#endif