#ifndef RECURSIVE_KMEANS_CU

#define RECURSIVE_KMEANS_CU

#include "gpu_ptr.h"
#include "defines.h"
#include "kernel-functions/align-memory.cu"
#include "kernel-functions/kmeans-utils.cu"

#include <thrust/device_ptr.h>
#include <thrust/sort.h>

void sort_labels_indexes(int* d_labels, int* d_indexes, size_t n) {
    auto keys   = thrust::device_pointer_cast(d_labels);
    auto values = thrust::device_pointer_cast(d_indexes);

    thrust::sort_by_key(keys, keys + n, values);
}

__global__
void split_clusters_kernel(
    int* clusters_starts,
    int* clusters_sizes,
    int n_finished_clusters,
    int max_bucket_size
){
    __shared__ int n_new_clusters;
    if(threadIdx.x == 0) {
        n_new_clusters = n_finished_clusters;
    }
    __syncthreads();

    for(int cluster_id = threadIdx.x;
            cluster_id < n_finished_clusters; 
            cluster_id += blockDim.x) {
        int start = clusters_starts[cluster_id];
        int size  = clusters_sizes[cluster_id];

        if(size > max_bucket_size) {
            int new_clusters = size / max_bucket_size + 1;
            int mean_size    = (size+new_clusters-1) / new_clusters;
            clusters_sizes[cluster_id] = mean_size;
            for(int i = 1; i < new_clusters; ++i) {
                int new_cluster_id = atomicAdd(&n_new_clusters, 1);
                clusters_starts[new_cluster_id] = start + i * mean_size;
                if(i == new_clusters - 1) {
                    clusters_sizes[new_cluster_id] = size - i * mean_size;
                } else {
                    clusters_sizes[new_cluster_id] = mean_size;
                }
            }
        }
    }

    //debug: print all clusters starts and sizes
    // __syncthreads();
    // if(threadIdx.x == 0) {
    //     printf("Total clusters after split: %d maxbucketsize: %d\n", n_new_clusters, max_bucket_size);
    // }
    // for(int cluster_id = threadIdx.x;
    //         cluster_id < n_new_clusters; 
    //         cluster_id += blockDim.x) {
    //     printf("Cluster %d start= %d size= %d\n", cluster_id, clusters_starts[cluster_id], clusters_sizes[cluster_id]);
    // }

}

__global__
void copy_and_pad_kernel(
    int* indexes,
    int* d_nodes_bucket,
    int* d_bucket_size,
    int* clusters_starts,
    int* clusters_sizes,
    int n_total_clusters,
    int max_bucket_size
){
    for(int cluster_id = blockIdx.x;
            cluster_id < n_total_clusters; 
            cluster_id += gridDim.x) {
        int start = clusters_starts[cluster_id];
        int size  = clusters_sizes[cluster_id];

        // Copy to bucket
        for(int i = threadIdx.x; i < size; i += blockDim.x) {
            d_nodes_bucket[cluster_id * max_bucket_size + i] = indexes[start + i];
        
            //debug: threadIdx.x == 0 && blockIdx.x == 0 prints cluster_id, i, start, size, indexes[start + i]
            // if(threadIdx.x == 0 && blockIdx.x == 0) {
            //     printf("cluster_id: %d, i: %d, start: %d, size: %d, index: %d\n", cluster_id, i, start, size, indexes[start + i]);
            // }
        }

        // Pad the rest with -1
        for(int i = threadIdx.x + size; i < max_bucket_size; i += blockDim.x) {
            d_nodes_bucket[cluster_id * max_bucket_size + i] = -1;
        }

        // Set bucket size
        if(threadIdx.x == 0) {
            d_bucket_size[cluster_id] = size;
        }
    }
    //debug: print first and last bucket
    // if(blockIdx.x == 0 && threadIdx.x == 0) {
    //     printf("First bucket size: %d\n", d_bucket_size[0]);
    //     printf("first bucket: ");
    //     for(int i = 0; i < max_bucket_size; ++i) {
    //         printf("%d ", d_nodes_bucket[i]);
    //     }
    //     printf("Last bucket size: %d\n", d_bucket_size[n_total_clusters - 1]);
    //     printf("last bucket: ");
    //     for(int i = 0; i < max_bucket_size; ++i) {
    //         printf("%d ", d_nodes_bucket[(n_total_clusters - 1) * max_bucket_size + i]);
    //     }
    // }

}

//////////////////////
// RECURSIVE KMEANS //
//////////////////////
void recursive_call(
    int* indexes,
    float* points,
    int dim,
    float* centroids,
    int* labels,
    float* dist_to_centroids,

    int n_points_subset,
    int k,
    int max_depth,
    int max_points,
    int depth,
    int my_offset,

    int* clusters_starts,
    int* clusters_sizes,
    int* finished_clusters, 
    int* not_finished_clusters
){
    // Base case: check depth and number of points
    if (depth >= max_depth) {
        // Mark cluster as finished
        int finished_index = (*finished_clusters)++;
        clusters_starts[finished_index] = my_offset;
        clusters_sizes[finished_index] = n_points_subset;

        (*not_finished_clusters) += n_points_subset / max_points;

        // printf("Max depth reached at depth %d with %d points.\n", depth, n_points_subset);

        return;
    }

    kmeanspp<true>(
        points,
        n_points_subset,
        dim,
        k,
        0,
        centroids,
        (uint*) labels,
        dist_to_centroids,
        indexes
    );

    // Count labels
    GpuPtr<int> this_labels_counts(k);
    cudaMemset(this_labels_counts.ptr(), 0, k * sizeof(int));

    //--- Launch kernel to count labels
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    int max_threads = deviceProp.maxThreadsPerBlock;
    int nthreads = max_threads/2;
    int nblocks = deviceProp.multiProcessorCount*(max_threads/nthreads);
    int shared_mem_size = nthreads * sizeof(int);
    int max_shared_mem = deviceProp.sharedMemPerBlock;
    if(shared_mem_size > max_shared_mem) {
        printf("Warning: shared memory size (%d) is larger than the device max shared memory per block (%d).", shared_mem_size, max_shared_mem);
    }
    count_labels_kernel<<<nblocks, nthreads, shared_mem_size>>>((uint*)labels, this_labels_counts.ptr(), n_points_subset, k);
    cudaDeviceSynchronize();

    // sort out indexes for each cluster
    sort_labels_indexes(labels, indexes, n_points_subset);

    int counts_host[k];
    cudaMemcpy(counts_host, this_labels_counts.ptr(), k * sizeof(int), cudaMemcpyDeviceToHost);

    // For each cluster, check if we need to recurse
    int offset = 0;
    for (int cluster_id = 0; cluster_id < k; ++cluster_id) {
        int count = counts_host[cluster_id];
        if (count >= max_points) {
            // Recurse on this cluster
            recursive_call(
                indexes + offset,
                points,
                dim,
                centroids,
                labels + offset,
                dist_to_centroids + offset,

                count,
                // k,
                // min(k * 2, 1024), // double k each recursion
                min((int)pow(2, depth + 1),1024), // double k each recursion), 
                // max(k / 2, 8), // half k each recursion
                // max(min(k,(count+max_points-1)/max_points),2),
                max_depth,
                max_points,
                depth + 1,
                my_offset + offset,

                clusters_starts,
                clusters_sizes,
                finished_clusters,
                not_finished_clusters
            );
        } else {
            // Mark cluster as finished
            int finished_index = (*finished_clusters)++;
            clusters_starts[finished_index] = my_offset + offset;
            clusters_sizes[finished_index] = count;
        }
        offset += count;
    }
}


TreeInfo recursive_kmeans(
    float* points,
    int n_points,
    int dim,
    int k,
    int max_depth,
    int max_bucket_size
){

    int finished_clusters = 0;
    int not_finished_clusters = 0;
    
    int* clusters_starts = (int*)malloc(n_points * sizeof(int));
    int* clusters_sizes  = (int*)malloc(n_points * sizeof(int));

    GpuPtr<int> indexes(n_points);
    indexes.fillSequential();

    // GpuPtr<float> centroids(k * dim);
    GpuPtr<float> centroids(1024 * dim);
    GpuPtr<int> labels(n_points);
    GpuPtr<float> dist_to_centroids(n_points);

    recursive_call(
        indexes.ptr(),
        points,
        dim,
        centroids.ptr(),
        labels.ptr(),
        dist_to_centroids.ptr(),

        n_points,
        k,
        // 2,
        max_depth,
        max_bucket_size,
        0,
        0,

        clusters_starts,
        clusters_sizes,
        &finished_clusters,
        &not_finished_clusters
    );

    int total_buckets = finished_clusters + not_finished_clusters;

    //debug print finished and not finished clusters
    printf("Finished clusters: %d\n", finished_clusters);
    printf("Not finished clusters: %d\n", not_finished_clusters);

    thrust::device_vector<int> d_nodes_bucket(total_buckets * max_bucket_size, -1);
    thrust::device_vector<int> d_bucket_size(total_buckets, 0);

    GpuPtr<int> clusters_starts_gpu(total_buckets);
    GpuPtr<int> clusters_sizes_gpu(total_buckets);
    clusters_starts_gpu.copyFromHost(clusters_starts, finished_clusters);
    clusters_sizes_gpu.copyFromHost(clusters_sizes, finished_clusters);

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    int max_threads = deviceProp.maxThreadsPerBlock;
    int nthreads = max_threads/2;
    int nblocks = deviceProp.multiProcessorCount*(max_threads/nthreads);

    split_clusters_kernel<<<1, max_threads>>>(
        clusters_starts_gpu.ptr(),
        clusters_sizes_gpu.ptr(),
        finished_clusters,
        max_bucket_size
    );

    copy_and_pad_kernel<<<nblocks, nthreads>>>(
        indexes.ptr(),
        thrust::raw_pointer_cast(d_nodes_bucket.data()),
        thrust::raw_pointer_cast(d_bucket_size.data()),
        clusters_starts_gpu.ptr(),
        clusters_sizes_gpu.ptr(),
        total_buckets,
        max_bucket_size
    );

    cudaDeviceSynchronize();

    TreeInfo tinfo = TreeInfo(total_buckets, max_bucket_size,
                              d_nodes_bucket, d_bucket_size);


    // tinfo.print_info();
    // tinfo.print_buckets();

    free(clusters_starts);
    free(clusters_sizes);

    return tinfo;
}

#endif // RECURSIVE_KMEANS_CU