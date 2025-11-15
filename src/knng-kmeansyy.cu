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

__global__ void create_contiguous_array(
    int* array
)
{
    // Set array with 0, 1, 2, ..., N-1
    int tid = blockDim.x*blockIdx.x + threadIdx.x;
    array[tid] = tid;

    __syncthreads();
}

__global__ void count_points_per_cluster(
    uint* sorted_labels,
    int N,
    int* label_counts
)
{
    //Find the transition of values in sorted_labels to count how many points per cluster
    int tid = blockDim.x*blockIdx.x + threadIdx.x;
    if(tid < N-1){
        if(sorted_labels[tid] != sorted_labels[tid+1]){
            atomicAdd(&label_counts[sorted_labels[tid]], 1);
        }
    }
}

TreeInfo create_bucket_from_kmeansyy(
    thrust::device_vector<RSFK_typepoints> &device_points,
    int N, int D, int VERBOSE,
    ForestLog& forest_log,
    int total_buckets=64,
    int max_iter = 30,
    int check_method = 2,
    // printf("check_method: 0 -> until max it\n"),
    // printf("              1 -> by squared norm error\n"),
    // printf("              2 -> by number of reassingments (default)\n"),
    int tolerance = 0.01,
    int init_method = 1, //0 -> random, 1 -> kmeans++
    int t_groups = 32
    )
{
    forest_log.count_tree += 1;
    
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

    kmeansGpu(
            thrust::raw_pointer_cast(device_points.data()), 
            N, D, total_buckets,
            max_iter,
            check_method,
            tolerance,
            init_method,
            t_groups,
            VERBOSE,
            d_labels
    );


    // Pass d_labels to thrust
    thrust::device_vector<uint> device_labels(N);
    cudaMemcpy(thrust::raw_pointer_cast(device_labels.data()), d_labels, sizeof(uint)*N, cudaMemcpyDeviceToDevice);

    // Argsort to create bucket indexes
    // thrust::device_vector<std::intptr_t> indices(N);
    thrust::device_vector<int> indices(N);
    thrust::sequence(indices.begin(), indices.end());
    thrust::sort_by_key(
        device_labels.begin(), device_labels.end(),
        indices.begin()
    );

    // 

    thrust::host_vector<uint> h_labels = device_labels;
    thrust::host_vector<int> h_indices = indices;

    // for(int i = 0; i < N; i++){
    //     std::cout << "Point " << i << " label: " << h_labels[i] << std::endl;
    // }
    // for(int i = 0; i < N; i++){
    //     std::cout << "Point " << i << " index: " << h_indices[i] << std::endl;
    // }

    // // Print sorted labels
    // std::cout << "Sorted labels: " << std::endl;
    // for(int i = 0; i < N; i++){
    //     std::cout << h_labels[i] << " ";
    // }
    // std::cout << std::endl;

    // Iterate over sorted labels and count the maximum bucket size
    int max_bucket_size = 0;
    int current_label = -1;
    int current_count = 0;
    for(int i = 0; i < N; i++){
        if(h_labels[i] != current_label){
            if(current_count > max_bucket_size){
                max_bucket_size = current_count;
            }
            current_label = h_labels[i];
            current_count = 1;
        }
        else{
            current_count++;
        }
    }
    // Check last bucket
    if(current_count > max_bucket_size){
        max_bucket_size = current_count;
    }
    // std::cout << "Max bucket size: " << max_bucket_size << std::endl;
    
    // Create padded bucket array (each cluster with max_bucket_size)
    thrust::host_vector<int> h_nodes_bucket(total_buckets * max_bucket_size, -1);
    thrust::host_vector<int> h_bucket_size(total_buckets, 0);
    
    // Fill the buckets in host
    for(int i = 0; i < N; i++){
        int label = h_labels[i];
        int index = h_indices[i];
        int pos = h_bucket_size[label];
        h_nodes_bucket[label * max_bucket_size + pos] = index;
        h_bucket_size[label]++;
    }
    thrust::device_vector<int> d_nodes_bucket(total_buckets * max_bucket_size, -1);
    thrust::device_vector<int> d_bucket_size(total_buckets, 0);    

    // Print padded buckets
    #define DEBUG_BUCKETS 1
    #ifdef DEBUG_BUCKETS
    for(int i = 0; i < total_buckets; i++){
        std::cout << "Bucket " << i << " (size " << h_bucket_size[i] << "): ";
        for(int j = 0; j < max_bucket_size; j++){
            std::cout << h_nodes_bucket[i * max_bucket_size + j] << " ";
        }
        std::cout << std::endl;
    }
    #endif

    // Update ForestInfo for max_bucket_size
    

    // exit(0);

    // const thrust::device_vector< int > v{std::cbegin(init), std::cend(init)};

    // // optimization to avoid unnecessary initialization of index to zero
    // auto const seq_iter =
    //     thrust::make_counting_iterator(
    //         static_cast< std::intptr_t >(0));

    // thrust::device_vector< std::intptr_t > index{seq_iter,
    //                                              thrust::next(seq_iter, v.size())};
    
    // auto const v_ptr = v.data();

    // thrust::sort(
    //     index.begin(), index.end(),
    //     [v_ptr] __host__ __device__ (std::intptr_t left_idx, std::intptr_t right_idx)
    //     {
    //         return v_ptr[left_idx] < v_ptr[right_idx];
    //     });

    // thrust::copy(
    //     index.cbegin(), index.cend(),
    //     std::ostream_iterator< std::intptr_t >(std::cout, ", "));
    // std::cout << std::endl;

    err = cudaFree(d_labels);
    if (err != cudaSuccess){
        fprintf(stderr, "Failed to free device vector d_labels (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    

    


    
    TreeInfo tinfo = TreeInfo(total_buckets, max_bucket_size,
                              d_nodes_bucket, d_bucket_size);

    return tinfo;

}

#endif