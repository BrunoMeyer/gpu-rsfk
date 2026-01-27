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

TreeInfo build_treeinfo(
    int* clusters_starts,
    int* clusters_sizes,
    int finished_clusters,
    int not_finished_clusters,
    int* indexes,
    int n_points,
    int max_bucket_size
) {
    int total_buckets = finished_clusters + not_finished_clusters;

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
    int nthreads = max_threads / 2;
    int nblocks =
        deviceProp.multiProcessorCount * (max_threads / nthreads);

    split_clusters_kernel<<<1, max_threads>>>(
        clusters_starts_gpu.ptr(),
        clusters_sizes_gpu.ptr(),
        finished_clusters,
        max_bucket_size
    );

    copy_and_pad_kernel<<<nblocks, nthreads>>>(
        indexes,
        thrust::raw_pointer_cast(d_nodes_bucket.data()),
        thrust::raw_pointer_cast(d_bucket_size.data()),
        clusters_starts_gpu.ptr(),
        clusters_sizes_gpu.ptr(),
        total_buckets,
        max_bucket_size
    );

    cudaDeviceSynchronize();
    gpuErrchk( cudaPeekAtLastError() );

    return TreeInfo(
        total_buckets,
        max_bucket_size,
        d_nodes_bucket,
        d_bucket_size
    );
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

    TreeInfo tinfo = build_treeinfo(
        clusters_starts,
        clusters_sizes,
        finished_clusters,
        not_finished_clusters,
        indexes.ptr(),
        n_points,
        max_bucket_size
    );

    // tinfo.print_info();
    // tinfo.print_buckets();

    free(clusters_starts);
    free(clusters_sizes);

    return tinfo;
}

//====================================================================
// KMEANS RECURSIVE SPLITTING GPU RESOURCES VERSION
class KMeansStream {
public:
    bool initialized{false};

    //kernel launch parameters
    cudaStream_t stream{nullptr};
    int nthreads;
    int maxthreads;
    int maxblocks;
    int maxwarps;
    int max_shared_mem;

    //data pointers
    float* points;      // dataset N × D (not owned)
    int* indexes;     // indexes of points in the dataset (not owned)

    //parameters
    int max_points;
    int dim;
    int k;

    //working space
    int wsstart;
    int wssize;

    //outputs
    GpuPtr<int> labels_count;  // K

    //cpu output buffers
    int* clusters_sizes{nullptr}; // K

    // working memory: device memory allocated to run kmeans, 
    // but nothing is stored here after the call (all results are in )
    GpuPtr<int> workmem_labels;     // N 
    GpuPtr<float> workmem_centroids;   // K × D
    GpuPtr<float> workmem_dist_to_centroids;   // N
    GpuPtr<float> workmem_sec_dist_to_centroids;   // N
    GpuPtr<int> workmem_chosen_centroids; // K
    GpuPtr<int> workmem_candidates_to_nextcent; // max warps 
    GpuPtr<float> workmem_max_min_cent_dist;   // max warps

    //====================================================================
    // class constructors

    //default constructor
    KMeansStream() = default;

    void init(
        float* d_points,
        int* d_indexes,
        int max_points_,
        int dim_,
        int k_,
        int nthreads_=0,
        int maxthreads_=0,
        int maxblocks_=0
    ) {
        if(initialized) {
            printf("WARNING: KMeansStream already initialized.\n");
            return;
        }

        points = d_points;
        indexes = d_indexes;
        max_points = max_points_;
        dim = dim_;
        k = k_;
        wsstart = 0;
        wssize = 0;
        nthreads = nthreads_;
        maxthreads = maxthreads_;
        maxblocks = maxblocks_;
        
        //get device properties
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, 0);
        if(maxthreads == 0) maxthreads = deviceProp.maxThreadsPerBlock;
        if(nthreads == 0) nthreads = maxthreads / 2;
        if(maxblocks == 0) {
            maxblocks = deviceProp.multiProcessorCount * (maxthreads / nthreads);
        }
        max_shared_mem = deviceProp.sharedMemPerBlock;

        //shared memory size for counting labels
        int shared_mem_size = nthreads * sizeof(int);
        if(shared_mem_size > max_shared_mem) {
            printf("Warning: shared memory size (%d) is larger than the device max shared memory per block (%d).", shared_mem_size, max_shared_mem);
        }
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

        //allocate output memory on gpu
        labels_count.allocate(k);

        //allocate working memory on gpu
        workmem_labels.allocate(max_points);
        workmem_centroids.allocate(k * dim);
        workmem_dist_to_centroids.allocate(max_points);
        workmem_sec_dist_to_centroids.allocate(max_points);
        workmem_chosen_centroids.allocate(k);

        maxwarps = maxblocks * nthreads / 32;
        workmem_candidates_to_nextcent.allocate(maxwarps);
        workmem_max_min_cent_dist.allocate(maxwarps);
        
        clusters_sizes  = (int*)malloc(k * sizeof(int));
        
        initialized = true;
    }

    KMeansStream(
        float* d_points,
        int* d_indexes,
        int max_points_,
        int dim_,
        int k_,
        int nthreads=0,
        int maxthreads=0,
        int maxblocks=0
    ){
        init(
            d_points,
            d_indexes,
            max_points_,
            dim_,
            k_,
            nthreads,
            maxthreads,
            maxblocks
        );
    }

    //====================================================================
    // class methods

    void run_async(
            int wsstart,
            int wssize,
            int nblocks=-1,
            int nthreads=-1
    ) {
        this->wsstart = wsstart;
        this->wssize = wssize;

        if(wssize > max_points) {
            printf("Warning: wssize (%d) is larger than max_points (%d). Setting wssize to max_points.\n", wssize, max_points);
            this->wssize = max_points;
            //resize working memory (costly but needed)
            workmem_labels.resize(max_points);
            workmem_dist_to_centroids.resize(max_points);
            workmem_sec_dist_to_centroids.resize(max_points);
        }

        if(nblocks > maxblocks || nthreads > maxthreads) {
            int needed_warps = nblocks * nthreads / 32;
            if(needed_warps > maxwarps) {
                printf("Warning: needed working memory is larger than the allocated at the initialization time (%d warps > %d warps). Resizing working memory.\n", needed_warps, maxwarps);
                maxwarps = needed_warps;
                workmem_candidates_to_nextcent.resize(maxwarps);
                workmem_max_min_cent_dist.resize(maxwarps);
            }
        }

        if(nblocks == -1) nblocks = this->maxblocks;
        if(nthreads == -1) nthreads = this->nthreads;
        //--- Run kmeans++ to assign labels
        kmeanspp_workflow_async<true>(
            points,
            wssize,
            dim,
            k,
            //OUTPUTS
            workmem_centroids.ptr(),
            (uint*)workmem_labels.ptr(),
            workmem_dist_to_centroids.ptr(),
            workmem_sec_dist_to_centroids.ptr(),
            (uint*)workmem_chosen_centroids.ptr(),
            (uint*)workmem_candidates_to_nextcent.ptr(),
            workmem_max_min_cent_dist.ptr(),
            //KERNEL PARAMS
            maxthreads,
            nthreads,
            nblocks,
            //OPTIONAL
            indexes + wsstart,
            stream
        );

        //count labels
        cudaMemsetAsync(labels_count.ptr(), 0, k * sizeof(int), stream);

        //--- Launch kernel to count labels
        int shared_mem_size = nthreads * sizeof(int);
        count_labels_kernel<<<nblocks, nthreads, shared_mem_size, stream>>>(
            workmem_labels.ptr(), 
            labels_count.ptr(), 
            wssize, 
            k
        );

        // copy labells count to clusters_sizes
        cudaMemcpyAsync(
            clusters_sizes, 
            labels_count.ptr(), 
            k * sizeof(int), 
            cudaMemcpyDeviceToHost,
            stream
        );
    }

    void sync() {
        cudaStreamSynchronize(stream);

        //debug:
        if(wsstart + wssize > max_points) {
            printf("ERROR: OUT OF RANGE!!! wsstart (%d) + wssize (%d) > max_points (%d)\n", wsstart, wssize, max_points);
            exit(1);
        }
        
        // sort out indexes for each cluster
        sort_labels_indexes(workmem_labels.ptr(), indexes + wsstart, wssize);
    }

    void cpy_results(
        int* out_clusters_sizes
    ) {
        memcpy(
            out_clusters_sizes,
            clusters_sizes,
            k * sizeof(int)
        );
    }

    void sync_and_cpy_results(
        int* out_clusters_sizes
    ) {
        sync();
        cpy_results(
            out_clusters_sizes
        );
    }
    //====================================================================
    // class properties and destructor
    ~KMeansStream() {
        if (stream) cudaStreamDestroy(stream);
        if (clusters_sizes) free(clusters_sizes);
    }

    KMeansStream(const KMeansStream&) = delete;
    KMeansStream& operator=(const KMeansStream&) = delete;

    KMeansStream(KMeansStream&&) noexcept = default;
    KMeansStream& operator=(KMeansStream&&) noexcept = default;

};

void stream_recursive_call(
    std::vector<KMeansStream>& streams,
    int* indexes,
    float* points,
    int dim,
    int k,
    int max_depth,
    int max_bucket_size,
    int depth,
    int my_offset,

    int* upper_level_clusters_sizes,

    int* final_clusters_starts,
    int* final_clusters_sizes,
    int* finished_clusters, 
    int* not_finished_clusters,

    int upper_level_points
){
    // int n_streams = streams.size();
    // printf("Depth %d: processing %d clusters with %d streams.\n", depth, k, n_streams);

    // For each cluster in this level, check if we need to recurse
    int offset = my_offset;
    for (int cluster_id = 0; cluster_id < k; ++cluster_id) {
        int count = upper_level_clusters_sizes[cluster_id];

        int maxblocks = streams[cluster_id].maxblocks;
        // int maxpoints = streams[cluster_id].max_points;
        int nblocks = maxblocks;

        // int nblocks = round((double)count/upper_level_points * maxblocks);
        nblocks = min(nblocks, maxblocks);

        if (count >= max_bucket_size && depth < max_depth) {
            //run kmeans in the stream
            // int stream_id = cluster_id % n_streams;
            int stream_id = cluster_id;
            streams[stream_id].run_async(
                offset,
                count,
                nblocks
            );
        } else {
            // Mark cluster as finished
            int finished_index = (*finished_clusters)++;
            final_clusters_starts[finished_index] = offset;
            final_clusters_sizes[finished_index] = count;

            //since all clusters has to be smaller than max_bucket_size
            //the number of not finished clusters is count / max_bucket_size
            //Note: this happens when max depth is reached and cluster is still too big
            (*not_finished_clusters) += count / max_bucket_size;
        }
        offset += count;
    }

    //allocate memory for next level clusters sizes
    int* level_clusters_sizes = (int*)malloc(k * k * sizeof(int));

    //sync all streams and get 
    for (int cluster_id = 0; cluster_id < k; ++cluster_id) {
        int count = upper_level_clusters_sizes[cluster_id];
        if (count >= max_bucket_size && depth < max_depth) {
            // int stream_id = cluster_id % n_streams;
            int stream_id = cluster_id;
            streams[stream_id].sync_and_cpy_results(
                &level_clusters_sizes[cluster_id*k]
            );
        }
    }
    
    offset = my_offset;
    for (int cluster_id = 0; cluster_id < k; ++cluster_id) {
        int count = upper_level_clusters_sizes[cluster_id];
        if (count >= max_bucket_size && depth < max_depth) {
            stream_recursive_call(
                streams,
                indexes,
                points,
                dim,
                k,
                max_depth,
                max_bucket_size,
                depth + 1,
                offset,

                &level_clusters_sizes[cluster_id*k],

                final_clusters_starts,
                final_clusters_sizes,
                finished_clusters,
                not_finished_clusters,
                count
            );
        }
        offset += count;
    }
    free(level_clusters_sizes);
}

TreeInfo stream_recursive_kmeans(
    float* points,
    int n_points,
    int dim,
    int k,
    int max_depth,
    int max_bucket_size
){
    int finished_clusters = 0;
    int not_finished_clusters = 0;
    
    int* final_clusters_starts = (int*)malloc(n_points * sizeof(int));
    int* final_clusters_sizes  = (int*)malloc(n_points * sizeof(int));

    GpuPtr<int> indexes(n_points);
    indexes.fillSequential();

    std::vector<KMeansStream> streams(k);
    for(int i = 0; i < k; ++i) {
        streams[i].init(
            points,
            indexes.ptr(),
            n_points,
            dim,
            k
        );
    }

    //run the first kmeans on the whole dataset in the first stream
    streams[0].run_async(
        0,
        n_points
    );
    int* level_clusters_sizes  = (int*)malloc(k * sizeof(int));

    //wait for the first kmeans to finish
    streams[0].sync_and_cpy_results(
        level_clusters_sizes
    );

    stream_recursive_call(
        streams,
        indexes.ptr(),
        points,
        dim,
        k,
        max_depth,
        max_bucket_size,
        0,
        0,

        level_clusters_sizes,

        final_clusters_starts,
        final_clusters_sizes,
        &finished_clusters,
        &not_finished_clusters,
        n_points
    );

    TreeInfo tinfo = build_treeinfo(
        final_clusters_starts,
        final_clusters_sizes,
        finished_clusters,
        not_finished_clusters,
        indexes.ptr(),
        n_points,
        max_bucket_size
    );
    
    // tinfo.print_info();
    // tinfo.print_buckets();
    
    free(level_clusters_sizes);
    free(final_clusters_starts);
    free(final_clusters_sizes);

    return tinfo;
}


#endif // RECURSIVE_KMEANS_CU