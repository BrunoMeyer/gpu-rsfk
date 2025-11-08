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

#ifndef __KMEANSYY__CU
#define __KMEANSYY__CU

#include "include/rsfk.h"
#include "kmeans/kmeans.h"

__global__
void count_labels_mt(uint* labels, int N, int* label_counts, int K){
    int tid = blockDim.x*blockIdx.x + threadIdx.x;
    if(tid < N){
        int label = labels[tid];
        if(label >= 0 && label < K){
            atomicAdd(&label_counts[label], 1);
        }
    }
}

__global__
void find_max_bucket_size(int* label_counts, int K, int* max_bucket_size){
    __shared__ int sm_max_bucket_size;
    if(threadIdx.x == 0){
        sm_max_bucket_size = 0;
    }
    __syncthreads();

    for(int i = threadIdx.x; i < K; i += blockDim.x){
        if(label_counts[i] > sm_max_bucket_size){
            sm_max_bucket_size = label_counts[i];
        }
    }
    __syncthreads();

    if(threadIdx.x == 0){
        atomicMax(max_bucket_size, sm_max_bucket_size);
    }
}

__global__
void create_buckets_mt(    RSFK_typepoints* points,
                        uint* labels,
                        int* nodes_bucket,
                        int* bucket_size,
                        int N, int D, int max_bucket_size, 
                        int total_buckets){

    int tid = blockDim.x*blockIdx.x + threadIdx.x;
    if(tid < N){
        uint label = labels[tid];
        if(label < total_buckets){
            int pos = atomicAdd(&bucket_size[label], 1);
            nodes_bucket[label*max_bucket_size + pos] = tid;
        }
        else{
            printf("Warning: point %d has invalid label %d\n", tid, label);
        }
    }
}

TreeInfo create_bucket_from_kmeansyy(
    thrust::device_vector<RSFK_typepoints> &device_points,
    int N, int D, int VERBOSE,
    std::string run_name="out.png",
    int total_buckets=128,
    int max_iter = 30,
    int check_method = 2,
    // printf("check_method: 0 -> until max it\n"),
    // printf("              1 -> by squared norm error\n"),
    // printf("              2 -> by number of reassingments (default)\n"),
    int tolerance = 0.01,
    int init_method = 1, //0 -> random, 1 -> kmeans++
    int t_groups = 64
    )
{
    int devUsed = 0;
    cudaSetDevice(devUsed);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, devUsed);

    // Tries to ensure that there are at least two blocks per multiprocessor
    int nthreads = deviceProp.maxThreadsPerMultiProcessor / 2;
    if(nthreads > deviceProp.maxThreadsPerBlock) nthreads = deviceProp.maxThreadsPerBlock;
    int nblocks = deviceProp.multiProcessorCount*(deviceProp.maxThreadsPerMultiProcessor/nthreads);

    // These vectors don't need to be initialized, will be outputs
	uint *d_labels = NULL;
	cudaError_t err = cudaMalloc((void **)&d_labels, sizeof(uint)*N);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to allocate device vector d_labels (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	float *d_centroids = NULL;
	err = cudaMalloc((void **)&d_centroids, sizeof(uint)*N);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to allocate device vector d_centroids (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

    kmeansGpu( 
            thrust::raw_pointer_cast(device_points.data()), 
            N, D, total_buckets,
            max_iter,
            check_method, tolerance,
            init_method,
            t_groups,
            VERBOSE,
            d_labels,
            d_centroids
    );

    // create buckets 
    // allocate d_label_counts and initilize with 0
    int* d_label_counts = NULL;
    err = cudaMalloc((void **)&d_label_counts, sizeof(int)*total_buckets);
    if (err != cudaSuccess){
        fprintf(stderr, "Failed to allocate device vector d_label_counts (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    cudaMemset(d_label_counts, 0, sizeof(int)*total_buckets);



    int* d_max_bucket_size = NULL;
    err = cudaMalloc((void **)&d_max_bucket_size, sizeof(int));
    if (err != cudaSuccess){
        fprintf(stderr, "Failed to allocate device vector d_max_bucket_size (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    cudaMemset(d_max_bucket_size, 0, sizeof(int));


    int nb = (N/nthreads) + ((N % nthreads) ? 1 : 0);
    count_labels_mt<<<nb, nthreads>>>( 
        d_labels,
        N,
        d_label_counts,
        total_buckets
    );



    find_max_bucket_size<<<nthreads, nthreads>>>( 
        d_label_counts,
        total_buckets,
        d_max_bucket_size
    );
    int max_bucket_size;
    cudaMemcpy(&max_bucket_size, d_max_bucket_size, sizeof(int), cudaMemcpyDeviceToHost);




    int* d_nodes_buckets = NULL;
    err = cudaMalloc((void **)&d_nodes_buckets, sizeof(int)*max_bucket_size*total_buckets);
    if (err != cudaSuccess){
        fprintf(stderr, "Failed to allocate device vector d_nodes_buckets (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }




    int* d_bucket_sizes = NULL;
    err = cudaMalloc((void **)&d_bucket_sizes, sizeof(int)*total_buckets);
    if (err != cudaSuccess){
        fprintf(stderr, "Failed to allocate device vector d_bucket_sizes (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    cudaMemset(d_bucket_sizes, 0, sizeof(int)*total_buckets);

    // TODO: TROCAR PELO ALGORITMO MRBIN
    create_buckets_mt<<<nb, nthreads>>>( 
                        thrust::raw_pointer_cast(device_points.data()),
                        d_labels,
                        d_nodes_buckets,
                        d_bucket_sizes,
                        N, D, max_bucket_size,
                        total_buckets
    );
    cudaDeviceSynchronize();

    printf("PASSOU0: KMeansYY Bucket Creation Complete: %d buckets created with max size %d\n", total_buckets, 
        max_bucket_size);

	err = cudaFree(d_labels);
    err = cudaFree(d_label_counts);
    err = cudaFree(d_max_bucket_size);

    printf("PASSOU1: KMeansYY Bucket Creation Complete: %d buckets created with max size %d\n", total_buckets, max_bucket_size);

    // thrust::device_vector<int> thr_nodes_bucket(
    //     d_nodes_buckets, d_nodes_buckets + total_buckets*max_bucket_size);
    // thrust::device_vector<int> thr_bucket_size(
    //     d_bucket_sizes, d_bucket_sizes + max_bucket_size);

    thrust::device_ptr<int> dptr_nodes(d_nodes_buckets);
    thrust::device_ptr<int> dptr_sizes(d_bucket_sizes);

    thrust::device_vector<int> thr_nodes_bucket(
        dptr_nodes, dptr_nodes + total_buckets * max_bucket_size);

    thrust::device_vector<int> thr_bucket_size(
        dptr_sizes, dptr_sizes + total_buckets);


    printf("PASSOU2: KMeansYY Bucket Creation Complete: %d buckets created with max size %d\n", total_buckets, max_bucket_size);

    TreeInfo tinfo = TreeInfo(total_buckets, max_bucket_size,
                              thr_nodes_bucket, thr_bucket_size);

    printf("PASSOU3: KMeansYY Bucket Creation Complete: %d buckets created with max size %d\n", total_buckets, max_bucket_size);

    err = cudaFree(d_nodes_buckets);
    err = cudaFree(d_bucket_sizes);

    if (err != cudaSuccess){
        fprintf(stderr, "Failed to allocate device vector d_bucket_sizes (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    printf("PASSOU4: KMeansYY Bucket Creation Complete: %d buckets created with max size %d\n", total_buckets, max_bucket_size);

    return tinfo;
}

#endif