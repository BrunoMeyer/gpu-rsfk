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
#include "kmeans/chrono.c"
#include "kmeans/kmeanspp_logc.cu"

#include <vector>
#include <cassert>
#include <iostream>

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
void create_buckets_mt( RSFK_typepoints* points,
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
    int tid = blockDim.x*blockIdx.x + threadIdx.x;
    if(tid < N-1){
        if(sorted_labels[tid] != sorted_labels[tid+1]){
            atomicAdd(&label_counts[sorted_labels[tid]], 1);
        }
    }
}

// ============================================================================
// HOST HELPER: Enforce bucket size limit on *sorted* labels (host side)
// ============================================================================

struct BucketSplitResult {
    int max_bucket_size;  // largest bucket after all splits
    int total_buckets;    // final number of labels (max_label + 1)
};

// h_labels must be sorted by label
BucketSplitResult enforce_bucket_size_limit_host(
    thrust::host_vector<uint>& h_labels,
    int N,
    int bucket_size_limit,
    int VERBOSE = 1
)
{
    assert(N >= 0);
    assert(bucket_size_limit > 0);

    if (N == 0) {
        return {0, 0};
    }

    // 1. Find current max label so we can assign new unique labels
    uint max_label = 0;
    for (int i = 0; i < N; ++i) {
        if (h_labels[i] > max_label) {
            max_label = h_labels[i];
        }
    }

    struct Bucket {
        int start;      // inclusive
        int len;        // length of bucket
        uint label;     // label value
    };

    bool need_split = true;
    int max_bucket_size = 0;

    while (need_split) {
        need_split = false;

        std::vector<Bucket> buckets;
        buckets.reserve(N); // upper bound
        max_bucket_size = 0;

        // 2. Build buckets from contiguous equal labels
        int i = 0;
        while (i < N) {
            int  start = i;
            uint label = h_labels[start];
            int  j     = start + 1;

            while (j < N && h_labels[j] == label) {
                ++j;
            }
            int len = j - start;

            assert(len > 0);
            assert(start >= 0 && start + len <= N);

            buckets.push_back(Bucket{start, len, label});
            if (len > max_bucket_size) {
                max_bucket_size = len;
            }

            i = j;
        }

        // All buckets within the limit? We are done.
        if (max_bucket_size <= bucket_size_limit) {
            break;
        }

        // 3. Split every oversized bucket:
        //    move the last half of the bucket to a new label.
        for (const auto& b : buckets) {
            if (b.len <= bucket_size_limit) {
                continue;
            }

            need_split = true;

            int half = b.len / 2;
            if (half <= 0) {
                continue; // paranoia guard
            }

            uint new_label = ++max_label; // new unique label

            int start_second_half = b.start + (b.len - half);
            assert(start_second_half >= b.start);
            assert(start_second_half <  b.start + b.len);
            assert(b.start + b.len <= N);

            for (int idx = start_second_half; idx < b.start + b.len; ++idx) {
                h_labels[idx] = new_label;
            }
        }

        if (need_split && VERBOSE > 0){
            std::cerr << "[WARNING] Some buckets exceeded the size limit of "
                      << bucket_size_limit
                      << " and were split. New max bucket size is "
                      << max_bucket_size
                      << ", new total buckets is "
                      << (max_label + 1)
                      << ".\n";
        }
        // Loop again if needed: now h_labels has more labels, but is still
        // grouped contiguously by label (old_label or new_label).
    }

    BucketSplitResult res;
    res.max_bucket_size = max_bucket_size;
    res.total_buckets   = static_cast<int>(max_label) + 1; // labels assumed 0..max_label

    return res;
}

// ============================================================================
// RECURSIVE KMEANS
// ============================================================================
// void rec_kmeans(
//     KmeansInfo& kinfo,
//     int K, //                               number of clusters per call
//     int n_calls,
//     int max_calls,
//     int bucket_size_limit,
//     int& out_max_bucket_size,
//     int VERBOSE
// ){
//     kmeanspp(
//         kinfo.points.ptr(),
//         N, D, K,
//         VERBOSE,
//         kinfo.centroids.ptr(),
//         kinfo.labels.ptr(),
//         kinfo.dist_to_centroids.ptr()
//     );
//     // Count labels
//     kinfo.countLabels();
    
// }

// ============================================================================
// MAIN FUNCTION: create_bucket_from_yykmeans
// ============================================================================

TreeInfo create_bucket_from_yykmeans(
    thrust::device_vector<RSFK_typepoints> device_points,
    int N, int D, int VERBOSE,
    ForestLog& forest_log,
    int total_buckets=128,
    int bucket_size_limit =1024,
    KMeansInfo* kinfo = nullptr,
    int max_iter = 32,
    int check_method = 2,
    // 0 -> until max it
    // 1 -> by squared norm error
    // 2 -> by number of reassingments (default)
    int tolerance = 0.01,
    int init_method = 1, //0 -> random, 1 -> kmeans++
    int t_groups = 32
    )
{
    // Initial number of clusters for k-means

    forest_log.count_tree += 1;
    
    // int devUsed = 0;
    // cudaSetDevice(devUsed);
    // cudaDeviceProp deviceProp;
    // cudaGetDeviceProperties(&deviceProp, devUsed);

    // // Tries to ensure that there are at least two blocks per multiprocessor
    // int nthreads = deviceProp.maxThreadsPerMultiProcessor / 2;
    // if(nthreads > deviceProp.maxThreadsPerBlock) nthreads = deviceProp.maxThreadsPerBlock;
    // int nblocks = deviceProp.multiProcessorCount*(deviceProp.maxThreadsPerMultiProcessor/nthreads);

    bool own_kinfo = false;
    if(kinfo == nullptr){
        own_kinfo = true;
        kinfo = new KMeansInfo(thrust::raw_pointer_cast(device_points.data()), N, D, total_buckets);
    }

    // ------------------------------------------------------------------------
    //mesure time

    chronometer_t ch_kmeans;
    chrono_reset(&ch_kmeans);
    chrono_start(&ch_kmeans);


    #if KMEANS_METHOD == KMEANSPP_LOGC

        int n_buckets = 0;
        if(D > 96){
            kmeanspp_logc<true>( //true, because data is aligned and vetorized read is worthy
                kinfo->points.ptr(),
                N, kinfo->logic_dim, total_buckets,
                VERBOSE,
                kinfo->centroids.ptr(),
                kinfo->labels.ptr(),
                &n_buckets,
                bucket_size_limit
            );
            total_buckets = n_buckets;
        } else {
            kmeanspp_logc<false>( //false, because data is not aligned or vetorized read is not worthy
                thrust::raw_pointer_cast(device_points.data()),
                N, D, total_buckets,
                VERBOSE,
                kinfo->centroids.ptr(),
                kinfo->labels.ptr(),
                &n_buckets,
                bucket_size_limit
            );
            total_buckets = n_buckets;
        }
    #elif KMEANS_METHOD == KMEANSPP
        kmeanspp(
            kinfo->points.ptr(),
            N, kinfo->logic_dim, total_buckets,
            VERBOSE,
            kinfo->centroids.ptr(),
            kinfo->labels.ptr(),
            kinfo->dist_to_centroids.ptr()
        );
    #elif KMEANS_METHOD == FULL_KMEANS 
        // Run k-means on GPU, labels are written into d_labels (0..total_buckets-1)
        kmeansGpu(
                thrust::raw_pointer_cast(device_points.data()),
                N, D, total_buckets,
                max_iter,
                check_method,
                tolerance,
                init_method,
                t_groups,
                3,
                kinfo->labels.ptr(),
                kinfo->centroids.ptr(),
                kinfo->dist_to_centroids.ptr()
                ,kinfo->points.ptr()
        );
    #elif KMEANS_METHOD == RECURSIVE_KMEANS
        
    #endif

    chrono_stop(&ch_kmeans);
    double kmeans_sec = (double)chrono_gettotal(&ch_kmeans)/(1000*1000*1000); 

    if(VERBOSE > 2)
        printf("KMeans time: %.6f sec\n", kmeans_sec);

    // Get labels
    uint* d_labels = kinfo->labels.ptr();

    // Pass d_labels to thrust
    // ------------------------------------------------------------------------
    // Step 1: copy device labels to thrust device_vector and argsort by label
    // ------------------------------------------------------------------------
    chronometer_t ch_argsort;
    chrono_reset(&ch_argsort);
    chrono_start(&ch_argsort);
            
    thrust::device_vector<uint> device_labels(N);
    
    cudaError_t err = cudaMemcpy(
        thrust::raw_pointer_cast(device_labels.data()),
        d_labels,
        static_cast<size_t>(N) * sizeof(uint),
        cudaMemcpyDeviceToDevice
    );
    assert(err == cudaSuccess);

    // Argsort: indices = [0, 1, ..., N-1], then sort labels and permute indices
    thrust::device_vector<int> indices(N);
    thrust::sequence(indices.begin(), indices.end());

    thrust::sort_by_key(
        device_labels.begin(), device_labels.end(),
        indices.begin()
    );

    // Move sorted labels & indices to host
    thrust::host_vector<uint> h_labels  = device_labels;
    thrust::host_vector<int>  h_indices = indices;

    chrono_stop(&ch_argsort);
    double argsort_sec = (double)chrono_gettotal(&ch_argsort)/(1000*1000*1000);
    if(VERBOSE > 2)
        printf("Argsort time: %.6f sec\n", argsort_sec);

    // ------------------------------------------------------------------------
    // Step 2: Enforce bucket size limit on sorted labels (host side)
    // ------------------------------------------------------------------------
    chronometer_t ch_enforce;
    chrono_reset(&ch_enforce);
    chrono_start(&ch_enforce);

    BucketSplitResult split_res = enforce_bucket_size_limit_host(
        h_labels,
        N,
        bucket_size_limit
    );

    int max_bucket_size = split_res.max_bucket_size;
    total_buckets       = split_res.total_buckets;  // IMPORTANT: update with new label count

    // Basic sanity: labels must be < total_buckets
    #ifdef DEBUG_BUCKETS
    {
        uint max_label_check = 0;
        for (int i = 0; i < N; ++i) {
            if (h_labels[i] > max_label_check) max_label_check = h_labels[i];
        }
        if (static_cast<int>(max_label_check) + 1 != total_buckets) {
            std::cerr << "[WARNING] Inconsistent labels: max_label=" 
                      << max_label_check << " total_buckets=" << total_buckets << std::endl;
        }
    }
    #endif

    // Optional: if you want to update d_labels too (in original, you didn't
    // actually need the unsorted labels anymore, so we can skip it safely).
    // If someday you DO need the updated labels on device in original order,
    // just uncomment this block:

    /*
    // Copy modified sorted labels back to device
    device_labels = h_labels;

    // Unsort: scatter back to original order using indices
    thrust::device_vector<uint> device_labels_unsorted(N);
    thrust::scatter(
        device_labels.begin(), device_labels.end(),
        indices.begin(),
        device_labels_unsorted.begin()
    );

    // Copy back to original device memory
    err = cudaMemcpy(
        d_labels,
        thrust::raw_pointer_cast(device_labels_unsorted.data()),
        static_cast<size_t>(N) * sizeof(uint),
        cudaMemcpyDeviceToDevice
    );
    assert(err == cudaSuccess);
    */

    chrono_stop(&ch_enforce);
    double enforce_sec = (double)chrono_gettotal(&ch_enforce)/(1000*1000*1000);
    if(VERBOSE > 2)
        printf("Enforce bucket size time: %.6f sec\n", enforce_sec);

    // ------------------------------------------------------------------------
    // Step 3: Build padded bucket array on HOST
    //         (size: total_buckets x max_bucket_size)
    // ------------------------------------------------------------------------
    chronometer_t ch_build_buckets;
    chrono_reset(&ch_build_buckets);
    chrono_start(&ch_build_buckets);

    thrust::host_vector<int> h_nodes_bucket(total_buckets * max_bucket_size, -1);
    thrust::host_vector<int> h_bucket_size(total_buckets, 0);

    for (int i = 0; i < N; ++i) {
        int label = static_cast<int>(h_labels[i]);
        int index = h_indices[i];

        // Safety checks
        assert(label >= 0 && label < total_buckets);

        int pos = h_bucket_size[label];
        assert(pos >= 0 && pos < max_bucket_size); // enforce_bucket_size_limit guarantees this

        h_nodes_bucket[label * max_bucket_size + pos] = index;
        h_bucket_size[label]++;
    }


    // ------------------------------------------------------------------------
    // Step 4: Copy buckets to DEVICE
    // ------------------------------------------------------------------------
    thrust::device_vector<int> d_nodes_bucket(total_buckets * max_bucket_size, -1);
    thrust::device_vector<int> d_bucket_size(total_buckets, 0);    

    cudaMemcpy(thrust::raw_pointer_cast(d_nodes_bucket.data()), h_nodes_bucket.data(),
                sizeof(int)*total_buckets*max_bucket_size, cudaMemcpyHostToDevice);
    cudaMemcpy(thrust::raw_pointer_cast(d_bucket_size.data()), h_bucket_size.data(),
                sizeof(int)*total_buckets, cudaMemcpyHostToDevice);


    chrono_stop(&ch_build_buckets);
    double build_buckets_sec = (double)chrono_gettotal(&ch_build_buckets)/(1000*1000*1000);
    // ------------------------------------------------------------------------
    // Cleanup
    // ------------------------------------------------------------------------
    // DON'T need to free d_labels, it's managed by kinfo
    // err = cudaFree(d_labels);
    // if (err != cudaSuccess){
    //     fprintf(stderr, "Failed to free device vector d_labels (error code %s)!\n", cudaGetErrorString(err));
    //     exit(EXIT_FAILURE);
    // }

    // Update ForestInfo for max_bucket_size

    // Health check and debug prints
    // thrust::host_vector<int> h_test_bucket_size = d_bucket_size;
    // //print bucket sizes
    // for(int i = 0; i < total_buckets; i++){
    //     std::cout << "Bucket " << i << " size: " << h_test_bucket_size[i] << std::endl;
    // }


    // int total_leaves = 0;
    // for(int i = 0; i < total_buckets; i++){
    //     if(h_test_bucket_size[i] > 0){
    //         total_leaves++;
    //     }
    // }
    // if(VERBOSE > 0){
    //     std::cout << "KMeans created " << total_leaves << " non-empty buckets out of " << total_buckets << " total buckets." << std::endl;
    //     std::cout << "Maximum bucket size is " << max_bucket_size << std::endl;
    // }

    //TOTAL TIME
    if(VERBOSE > 2){    
        double total_sec = kmeans_sec + argsort_sec + enforce_sec + build_buckets_sec;
        printf("-------------------------------------\n");
        printf("Total bucket creation time: %.6f sec\n", total_sec);
        printf("-------------------------------------\n");
    }
    if(own_kinfo){
        delete kinfo;
    }

    // ------------------------------------------------------------------------
    // Build TreeInfo with final total_buckets and max_bucket_size
    // ------------------------------------------------------------------------
    TreeInfo tinfo = TreeInfo(total_buckets, max_bucket_size,
                              d_nodes_bucket, d_bucket_size);

    return tinfo;
}

#endif
