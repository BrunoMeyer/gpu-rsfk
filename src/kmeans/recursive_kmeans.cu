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
            int new_clusters = (size + max_bucket_size - 1) / max_bucket_size;
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
        // printf("cluster_id: %d, start: %d, size: %d\n", cluster_id, start, size);

        // Copy to bucket
        for(int i = threadIdx.x; i < size; i += blockDim.x) {
            d_nodes_bucket[cluster_id * max_bucket_size + i] = indexes[start + i];
        
            //debug: threadIdx.x == 0 && blockIdx.x == 0 prints cluster_id, i, start, size, indexes[start + i]
            // if(threadIdx.x == 0 && i+start < 100000) {
            //     // printf("cluster_id: %d, i: %d, start: %d, size: %d\n", cluster_id, i, start, size);
            //     printf("cluster_id: %d, i: %d, start: %d, size: %d, index: %d\n", cluster_id, i, start, size, indexes[start + i]);
            // }
        }

        // Pad the rest with -1
        for(int i = threadIdx.x + size; i < max_bucket_size; i += blockDim.x) {
            d_nodes_bucket[cluster_id * max_bucket_size + i] = -1;
        }

        // // Set bucket size
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

// __global__
// void copy_and_remove_padding_kernel(
//     int* d_nodes_bucket,
//     int* d_bucket_size,
//     int* indexes,
//     int* clusters_starts,
//     int n_total_clusters,
//     int max_bucket_size
// ){
//     __shared__ int size_sh;
//     if(threadIdx.x == 0) {
//         size_sh = 0;
//     }
//     for(int cluster_id = blockIdx.x;
//             cluster_id < n_total_clusters; 
//             cluster_id += gridDim.x) {
//         __syncthreads();
//         int bsize = d_bucket_size[cluster_id];
//         int csize = 0;

//         // Copy from bucket
//         for(int i = threadIdx.x; i < bsize && i < max_bucket_size; i += blockDim.x) {
//             int idx = d_nodes_bucket[cluster_id * max_bucket_size + i];
//             if(idx != -1) {
//                 indexes[clusters_starts[cluster_id] + i] = idx;
//                 csize++;
//             }
//         }

//         // Set cluster size
//         atomicAdd(&size_sh, csize);
//         __syncthreads();
//         if(threadIdx.x == 0) {
//             d_bucket_size[cluster_id] = size_sh;
//             size_sh = 0;
//         }
//     }
// }

TreeInfo build_treeinfo(
    int* clusters_starts,
    int* clusters_sizes,
    int finished_clusters,
    int not_finished_clusters,
    int* d_indexes,
    int max_bucket_size,
    int verbose = 0
) {

    int total_buckets = finished_clusters + not_finished_clusters;

    if(verbose){
        printf("Finished clusters: %d\n", finished_clusters);
        printf("Not finished clusters: %d\n", not_finished_clusters);
    }

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

    // cudaDeviceSynchronize();
    // gpuErrchk( cudaPeekAtLastError() );

    // debug: print all clusters starts and sizes after split
    // int* h_clusters_starts = (int*)malloc(total_buckets * sizeof(int));
    // int* h_clusters_sizes  = (int*)malloc(total_buckets * sizeof(int));
    // // int* h_indexes = (int*)malloc(total_buckets * max_bucket_size * sizeof(int));
    // clusters_starts_gpu.copyToHost(h_clusters_starts, total_buckets);
    // clusters_sizes_gpu.copyToHost(h_clusters_sizes, total_buckets);
    // printf("clusters_sizes_gpu.size() = %zu\n",clusters_sizes_gpu.size());
    // // cudaMemcpy(
    // //     h_indexes,
    // //     d_indexes,
    // //     total_buckets * max_bucket_size * sizeof(int),
    // //     cudaMemcpyDeviceToHost
    // // );
    // for(int i = 0; i < total_buckets; ++i) {
    //     printf("Cluster %d start= %d size= %d\n", i, h_clusters_starts[i], h_clusters_sizes[i]);
    // }
    // printf("d_indexes = %p\n", d_indexes);
    // cudaDeviceSynchronize();
    // gpuErrchk( cudaPeekAtLastError() );

    copy_and_pad_kernel<<<nblocks, nthreads>>>(
        d_indexes,
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
    int nthreads{0};
    int maxthreads{0};
    int maxblocks{0};
    int maxwarps{0};
    int max_shared_mem{0};

    //data pointers
    float* points{nullptr};      // dataset N × D (not owned)
    int* indexes{nullptr};     // indexes of points in the dataset (not owned)

    //parameters
    int n_points{0};
    int dim{0};
    int k{0};

    //working space
    int wsstart{0};
    int wssize{0};

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
        int n_points_,
        int dim_,
        int k_,
        int nthreads_=0
    ) {
        if(initialized) {
            printf("WARNING: KMeansStream already initialized.\n");
            return;
        }

        points = d_points;
        indexes = d_indexes;
        n_points = n_points_;
        dim = dim_;
        k = k_;
        wsstart = 0;
        wssize = 0;
        nthreads = nthreads_;
        
        //get device properties
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, 0);
        maxthreads = deviceProp.maxThreadsPerBlock;
        if(nthreads == 0) nthreads = maxthreads / 2;
        maxblocks = deviceProp.multiProcessorCount * (maxthreads / nthreads);
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
        workmem_labels.allocate(n_points);
        workmem_centroids.allocate(k * dim);
        workmem_dist_to_centroids.allocate(n_points);
        workmem_sec_dist_to_centroids.allocate(n_points);
        workmem_chosen_centroids.allocate(k);

        maxwarps = maxblocks * nthreads / 32;
        workmem_candidates_to_nextcent.allocate(maxwarps);
        workmem_max_min_cent_dist.allocate(maxwarps);
        
        if(k > 0)
            clusters_sizes  = (int*)malloc(k * sizeof(int));
        
        initialized = true;
    }

    void update(
        float* new_points,
        int* new_indexes,
        int new_points_size,
        int new_dim,
        int new_k
    ){
        if(!initialized) {
            init(
                new_points,
                new_indexes,
                new_points_size,
                new_dim,
                new_k
            );
            return;
        }
        k = new_k;
        points = new_points;
        indexes = new_indexes;
        dim = new_dim;
        n_points = new_points_size;

        //resize working memory if needed
        if(new_points_size > workmem_labels.size()) {
            //all working memory dependent on number of points
            workmem_labels.resize(new_points_size);
            workmem_dist_to_centroids.resize(new_points_size);
            workmem_sec_dist_to_centroids.resize(new_points_size);
        }
        if(new_k > labels_count.size()) {
            labels_count.resize(new_k);
            if(clusters_sizes) free(clusters_sizes);
            clusters_sizes  = (int*)malloc(new_k * sizeof(int));
        }
        if(new_k * new_dim > workmem_centroids.size())
            workmem_centroids.resize(new_k * new_dim);
    }

    KMeansStream(
        float* d_points,
        int* d_indexes,
        int n_points,
        int dim,
        int k,
        int nthreads=0
    ){
        init(
            d_points,
            d_indexes,
            n_points,
            dim,
            k,
            nthreads
        );
    }

    //====================================================================
    // class methods

    void check_inputs(
            int wsstart,
            int wssize,
            int max_points,
            int nblocks,
            int nthreads,
            int verbose=0
        ) {
        if(points == nullptr) {
            printf("Error: KMeansStream not initialized with points.\n");
            exit(1);
        }
        if(indexes == nullptr) {
            printf("Error: KMeansStream not initialized with indexes.\n");
            exit(1);
        }


        if(wssize > max_points) {
            printf("Warning: wssize (%d) is larger than max_points (%d). Setting wssize to max_points.\n", wssize, max_points);
            this->wssize = max_points;
            //resize working memory (costly but needed)
            workmem_labels.resize(max_points);
            workmem_dist_to_centroids.resize(max_points);
            workmem_sec_dist_to_centroids.resize(max_points);
        }

        if(wsstart + wssize > max_points) {
            printf("Warning: wsstart + wssize (%d + %d = %d) is larger than max_points (%d). Adjusting wsstart to fit in max_points.\n", wsstart, wssize, wsstart + wssize, max_points);
            this->wsstart = max(0, max_points - wssize);
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
        if(verbose) {
            printf("KMeansStream inputs check:\n");
            printf(" wsstart: %d\n", this->wsstart);
            printf(" wssize: %d\n", this->wssize);
            printf(" max_points: %d\n", max_points);
        }
    }

    void run_async(
            int wsstart,
            int wssize,
            int nblocks=-1,
            int nthreads=-1
        ) {
        if(nblocks == -1) nblocks = this->maxblocks;
        if(nthreads == -1) nthreads = this->nthreads;

        this->wsstart = wsstart;
        this->wssize = wssize;


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
        // cudaStreamSynchronize(stream); 
        // gpuErrchk( cudaPeekAtLastError() );

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
        gpuErrchk( cudaPeekAtLastError() );
        
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
    int min_bucket_size,
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

    // Guarantee that all clusters have at least min_bucket_size elements.
    for (int cluster_id = 0; cluster_id < k; ++cluster_id) {
        int count = upper_level_clusters_sizes[cluster_id];
        if (count < min_bucket_size) {
            if (count == 0){
                continue;
            }
            // Join cluster i with cluster i + 1.
            for(int next_id = cluster_id+1; next_id < k; ++next_id){
                int next_count = upper_level_clusters_sizes[next_id];
                count+=next_count;
                upper_level_clusters_sizes[cluster_id]=count;
                upper_level_clusters_sizes[next_id]=0;
                if(count >= min_bucket_size) break;
            }
            if(count < min_bucket_size) {
                // If i cannot be joined with i + 1 (i + 1 > k), then join cluster i with i - 1.
                for(int previous_id = cluster_id-1; previous_id >= 0; --previous_id){
                    int previous_count = upper_level_clusters_sizes[previous_id];
                    count+=previous_count;
                    upper_level_clusters_sizes[previous_id]=count;
                    upper_level_clusters_sizes[previous_id+1]=0;
                    if(count >= min_bucket_size) break;
                }
                // If all clusters are joined, give up: the cluster is equal to the upper-level cluster.
                if(upper_level_clusters_sizes[0] == upper_level_points){
                    // Mark cluster as finished
                    int finished_index = (*finished_clusters)++;  
                    final_clusters_starts[finished_index] = my_offset;
                    final_clusters_sizes[finished_index] = upper_level_clusters_sizes[0];
                    
                    //since all clusters has to be smaller than max_bucket_size
                    //the number of not finished clusters is count / max_bucket_size
                    //Note: this happens when max depth is reached and cluster is still too big
                    (*not_finished_clusters) += upper_level_clusters_sizes[0] / max_bucket_size;
                    
                    printf("WARNING DEBUG: KMEANS PRODUCED THE SAME CLUSTERING TWICE\n");
                    return;
                }
                break;
            }
        }
    }
    
    // For each cluster in this level, check if we need to recurse
    int offset = my_offset;
    for (int cluster_id = 0; cluster_id < k; ++cluster_id) {
        int count = upper_level_clusters_sizes[cluster_id];

        int maxblocks = streams[cluster_id].maxblocks;
        // int maxpoints = streams[cluster_id].max_points;
        int nblocks = maxblocks;

        // int nblocks = round((double)count/upper_level_points * maxblocks);
        // nblocks = min(nblocks, maxblocks);

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
            if(count > 0) {   
                // Mark cluster as finished
                int finished_index = (*finished_clusters)++;  
                
                final_clusters_starts[finished_index] = offset;
                final_clusters_sizes[finished_index] = count;
                
                //since all clusters has to be smaller than max_bucket_size
                //the number of not finished clusters is count / max_bucket_size
                //Note: this happens when max depth is reached and cluster is still too big
                (*not_finished_clusters) += count / max_bucket_size;
            }
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
                min_bucket_size,
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

TreeInfo stream_recursive_kmeans_core(
    float* points,
    int n_points,
    int dim,
    int k,
    int max_depth,
    int min_bucket_size,
    int max_bucket_size,
    int* indexes,
    std::vector<KMeansStream>& streams,
    int* clusters_sizes
){

    int finished_clusters = 0;
    int not_finished_clusters = 0;
    
    int* final_clusters_starts = (int*)malloc(n_points * sizeof(int));
    int* final_clusters_sizes  = (int*)malloc(n_points * sizeof(int));

    stream_recursive_call(
        streams,
        indexes,
        points,
        dim,
        k,
        max_depth,
        min_bucket_size,
        max_bucket_size,
        0,
        0,

        clusters_sizes,

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
        indexes,
        max_bucket_size
    );
    
    // tinfo.print_info();
    // tinfo.print_buckets();
    free(final_clusters_starts);
    free(final_clusters_sizes);

    return tinfo;
}

TreeInfo stream_recursive_kmeans(
    float* points,
    int n_points,
    int dim,
    int k,
    int max_depth,
    int min_bucket_size,
    int max_bucket_size
){
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

    TreeInfo tinfo = stream_recursive_kmeans_core(
        points,
        n_points,
        dim,
        k,
        max_depth,
        min_bucket_size,
        max_bucket_size,
        indexes.ptr(),
        streams,
        level_clusters_sizes
    );
    
    free(level_clusters_sizes);

    return tinfo;
}

//====================================================================
// HIERARCHICAL HYBRID KMEANS CLASS
//====================================================================
//
//
//
//
//
class HierarchicalKMeans {
private:
    bool initialized{false};
    bool finished{false};

    std::vector<KMeansStream> streams;
    
    //gpu pointers
    GpuPtr<int> own_indexes; 
    int* indexes_ptr{nullptr}; // indexes of points in the dataset (not owned)
    int indexes_size{0};
    float* points{nullptr};

    int finished_clusters{0};
    int not_finished_clusters{0};

    //cpu output buffers
    int* final_clusters_starts{nullptr};
    int* final_clusters_sizes{nullptr};
    
    // //rsfk
    //
    bool rsfk_initialized{false};
    float* forest_log_output{nullptr};
    std::unique_ptr<RSFK> rsfk;
    std::unique_ptr<TreeInfo> rsfk_treeinfo;
    bool rsfk_finished{false};

    int* cpu_nodes_buckets{nullptr};
    int* cpu_bucket_sizes{nullptr};


public:
    int n_streams{0};
    int n_points{0};
    int max_points{0};
    int dim{0};
    int max_k{0};
    int max_bucket_size{0};
    
    //RSFK parameters
    int rsfk_min_bucket_size{0};
    int rsfk_max_bucket_size{0};
    int rsfk_max_depth{0};

    //kernel launch parameters
    int nthreads{0};
    int maxthreads{0};
    int maxblocks{0};
    int max_shared_mem{0};

    //default constructor
    HierarchicalKMeans() = default;

    bool is_initialized() {
        return initialized;
    }

    void init(
        float* points_,
        int n_points_,
        int dim_,
        int max_k_,
        int n_streams_,
        int rsfk_min_bucket_size_,
        int rsfk_max_bucket_size_,
        int rsfk_max_depth_

    ) {
        if(initialized) {
            printf("WARNING: HierarchicalKMeans already initialized.\n");
            return;
        }
        initialized = true;
        points = points_;
        n_points = n_points_;
        dim = dim_;
        max_k = max_k_;
        max_points = n_points_;

        //get device properties
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, 0);
        if(maxthreads == 0) maxthreads = deviceProp.maxThreadsPerBlock;
        if(nthreads == 0) nthreads = maxthreads / 2;
        if(maxblocks == 0) {
            maxblocks = deviceProp.multiProcessorCount * (maxthreads / nthreads);
        }
        max_shared_mem = deviceProp.sharedMemPerBlock;

        n_streams = n_streams_;
        streams.resize(n_streams);

        if(n_points > 0) {
            final_clusters_starts = (int*)malloc(n_points * sizeof(int));
            final_clusters_sizes  = (int*)malloc(n_points * sizeof(int));
        }

        init_rsfk(
            rsfk_min_bucket_size_,
            rsfk_max_bucket_size_,
            rsfk_max_depth_
        );
    }

    void set_params(
        int new_k,
        int new_n_streams,
        int* new_indexes,
        int new_indexes_size
    ){
        if(!initialized) {
            printf("ERROR: HierarchicalKMeans is not initialized.\n");
            return;
        }
        //resize working memory if needed
        if(new_n_streams != n_streams) {
            n_streams = new_n_streams;
            streams.resize(n_streams);
        }
        if(new_k < max_k)
            max_k = new_k;
        for(int i = 0; i < n_streams; ++i) {
            streams[i].update(
                points,
                new_indexes,
                new_indexes_size,
                dim,
                max_k
            );
        }
        n_streams = new_n_streams;
        indexes_size = new_indexes_size;
        indexes_ptr = new_indexes;
    }

    //destructor
    ~HierarchicalKMeans() {
        if(final_clusters_starts) free(final_clusters_starts);
        if(final_clusters_sizes) free(final_clusters_sizes);
        if(forest_log_output) free(forest_log_output);
        if(cpu_nodes_buckets) free(cpu_nodes_buckets);
        if(cpu_bucket_sizes) free(cpu_bucket_sizes);
    }

    void restart(){
        finished = false;
        finished_clusters = 0;
        not_finished_clusters = 0;
    }

    TreeInfo run_kmeans(
            int max_depth,
            int min_bucket_size,
            int max_bucket_size_,
            int k
    ) {

        if(!initialized) {
            printf("ERROR: HierarchicalKMeans not initialized.\n");
            exit(1);
        }
        restart();
        max_bucket_size = max_bucket_size_;        

        own_indexes.allocate(n_points);
        own_indexes.fillSequential();

        set_params(
            k,
            n_streams,
            own_indexes.ptr(),
            n_points
        );

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
            own_indexes.ptr(),
            points,
            dim,
            k,
            max_depth,
            min_bucket_size,
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

        free(level_clusters_sizes);

        finished = true;

        return build_treeinfo(
            final_clusters_starts,
            final_clusters_sizes,
            finished_clusters,
            not_finished_clusters,
            own_indexes.ptr(),
            max_bucket_size
        );
    }

    void init_rsfk(
        int min_cluster_size,
        int max_cluster_size,
        int max_depth
    ) {
        if(!rsfk_initialized) {
            forest_log_output = (float*)malloc(1000 * sizeof(float)); //1 thousand floats for logging
            rsfk_initialized = true;
        }
        if(min_cluster_size*2+1 > max_cluster_size) {
            printf("ERROR: rsfk_min_bucket_size*2+1 (%d) > rsfk_max_bucket_size (%d). Adjusting rsfk_max_bucket_size.\n", min_cluster_size*2+1, max_cluster_size);
            max_cluster_size = min_cluster_size*2+1;
        }

        rsfk = std::make_unique<RSFK>(
            points,
            nullptr,
            nullptr,
            nullptr,
            min_cluster_size,
            max_cluster_size,
            max_depth,
            0, //random state
            0,
            forest_log_output
        );

        rsfk_min_bucket_size = min_cluster_size;
        rsfk_max_bucket_size = max_cluster_size;
        rsfk_max_depth = max_depth;

    }

    bool is_rsfk_initialized() {
        return rsfk_initialized;
    }

    void check_rsfk(){
        if(!rsfk_initialized) {
            printf("ERROR: RSFK not initialized. Call init_rsfk() before run_rsfk().\n");
            return;
        }
        
        if(rsfk->points == nullptr) {
            printf("ERROR: RSFK points pointer is null.\n");
            exit(1);
        }

        if(points == nullptr) {
            printf("ERROR: HierarchicalKMeans points pointer is null.\n");
            exit(1);
        }
    }

    void run_rsfk(){
        check_rsfk();

        // printf("Running RSFK...\n");
        // ForestLog log(10);
        // thrust::device_vector<float> thrust_points(points, points + n_points * dim);
        // TreeInfo treeinfo = rsfk->create_bucket_from_sample_tree(
        //     thrust_points,
        //     n_points,
        //     dim,
        //     1, //verbose
        //     log,
        //     "hierarchical_kmeans_rsfk_run",
        //     false,
        //     nullptr
        // );
        // printf("RSFK produced %d buckets with max bucket size %d.\n", rsfk_treeinfo->total_leaves, rsfk_treeinfo->max_child);

        TreeInfo treeinfo = rsfk->cluster_by_sample_tree(
            n_points,
            dim,
            0,
            &cpu_nodes_buckets,
            &cpu_bucket_sizes,
            "hierarchical_kmeans_rsfk_run"
        );
        
        rsfk_treeinfo = std::make_unique<TreeInfo>(treeinfo);
        // rsfk_treeinfo->print_info();
        // printf("device_nodes_buckets size = %d\n", rsfk_treeinfo->device_nodes_buckets.size());
        // printf("device_bucket_sizes size = %d\n", rsfk_treeinfo->device_bucket_sizes.size());
        // treeinfo.print_info();
        // printf("device_nodes_buckets size = %zu\n", treeinfo.device_nodes_buckets.size());
        // printf("device_bucket_sizes size = %zu\n", treeinfo.device_bucket_sizes.size());


        rsfk_finished = true;
        // return tinfo;
    }
    
    TreeInfo create_bucket_kmeans(
        int max_depth,
        int min_bucket_size,
        int max_bucket_size_,
        int k
    ) {
        run_kmeans(
            max_depth,
            min_bucket_size,
            max_bucket_size_,
            k
        );
        return build_treeinfo(
            final_clusters_starts,
            final_clusters_sizes,
            finished_clusters,
            not_finished_clusters,
            own_indexes.ptr(),
            max_bucket_size
        );
    }

    TreeInfo run_hybrid(
        int max_depth,
        int min_bucket_size,
        int max_bucket_size,
        int k
    ) {
        restart();
        run_rsfk();

        int nclusters = rsfk_treeinfo->total_leaves;
        int maxclustersize = rsfk_treeinfo->max_child;
        // int* gpu_nodes_buckets = thrust::raw_pointer_cast(rsfk_treeinfo->device_nodes_buckets.data());
        // int* gpu_bucket_sizes = thrust::raw_pointer_cast(rsfk_treeinfo->device_bucket_sizes.data());
        GpuPtr<int> gpu_nodes_buckets;
        // GpuPtr<int> gpu_bucket_sizes;

        gpu_nodes_buckets.createFromHost(
            cpu_nodes_buckets,
            nclusters * maxclustersize
        );
        // gpu_bucket_sizes.createFromHost(
        //     cpu_bucket_sizes,
        //     nclusters
        // );
        // auto& v = rsfk_treeinfo->device_nodes_buckets;

        // std::cout << "size = " << v.size()
        //         << ", capacity = " << v.capacity()
        //         << std::endl;
        //printing device vector contents

        if(!gpu_nodes_buckets.ptr()) {
            printf("ERROR: RSFK did not produce valid GPU buckets.\n");
            exit(1);
        }

        set_params(
            k,
            n_streams,
            gpu_nodes_buckets.ptr(),
            nclusters*maxclustersize
        );

        int* level_clusters_sizes  = (int*)malloc(k * sizeof(int));
        for(int i = 0; i < nclusters; ++i) {
            int count = cpu_bucket_sizes[i];
            int offset = i*maxclustersize;
            if (count < max_bucket_size) {
                if(count > 0) {   
                    // Mark cluster as finished
                    int finished_index = finished_clusters++;  
                    
                    final_clusters_starts[finished_index] = offset;
                    final_clusters_sizes[finished_index] = count;
                    
                    //since all clusters has to be smaller than max_bucket_size
                    //the number of not finished clusters is count / max_bucket_size
                    //Note: this happens when max depth is reached and cluster is still too big
                    not_finished_clusters += count / max_bucket_size;
                }
                continue;
            }
            // streams[0].check_inputs(
            //     i*maxclustersize,
            //     cpu_bucket_sizes[i],
            //     nclusters*maxclustersize,
            //     streams[0].maxblocks,
            //     streams[0].nthreads,
            //     1
            // );
            //run the first kmeans on the whole bucket in the first stream
            streams[0].run_async(
                i*maxclustersize,
                cpu_bucket_sizes[i]
            );

            // //wait for the first kmeans to finish
            streams[0].sync_and_cpy_results(
                level_clusters_sizes
            );

            stream_recursive_call(
                streams,
                gpu_nodes_buckets.ptr(),
                points,
                dim,
                k,
                max_depth,
                min_bucket_size,
                max_bucket_size,
                0,
                i*maxclustersize,
                level_clusters_sizes,
                final_clusters_starts,
                final_clusters_sizes,
                &finished_clusters,
                &not_finished_clusters,
                cpu_bucket_sizes[i]
            );
        }
        free(level_clusters_sizes);
        finished = true;
        // printf("Hybrid KMeans finished with %d finished clusters and %d not finished clusters.\n", finished_clusters, not_finished_clusters);
        
        // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        // debug
        // check final clusters starts and sizes
        // int debug = 0;
        // if(debug){
        //     int rsfk_bucket_size = nclusters*maxclustersize;
        //     for(int i = 0; i < finished_clusters; ++i) {
        //         int start = final_clusters_starts[i];
        //         int size = final_clusters_sizes[i];
        //         if(start < 0 || start >= rsfk_bucket_size) {
        //             printf("ERROR: final_clusters_starts[%d] = %d is out of bounds (0, %d)\n", i, start, rsfk_bucket_size-1);
        //             // exit(1);
        //         }
        //         if(size <= 0 || start + size > rsfk_bucket_size) {
        //             printf("ERROR: INVALID SIZE! final_clusters_sizes[%d] = %d is invalid for start %d (size must be > 0 and start + size <= %d)\n", i, size, start, rsfk_bucket_size);
        //             // exit(1);
        //         }
        //     }
        //     // print final clusters info
        //     printf("Final clusters info:\n");
        //     // sort final clusters by start index
        //     std::vector<std::pair<int, int>> clusters_info;
        //     for(int i = 0; i < finished_clusters; ++i) {
        //         clusters_info.push_back(std::make_pair(final_clusters_starts[i], final_clusters_sizes[i]));
        //     }
        //     std::sort(clusters_info.begin(), clusters_info.end());
        //     //check for overlapping clusters or gaps
        //     if(clusters_info[0].first != 0) {
        //         printf("ERROR: first cluster does not start at 0, starts at %d\n", clusters_info[0].first);
        //         // exit(1);
        //     }
        //     if(clusters_info[finished_clusters-1].first + clusters_info[finished_clusters-1].second != rsfk_bucket_size) {
        //         printf("ERROR: last cluster does not end at rsfk_bucket_size, ends at %d\n", clusters_info[finished_clusters-1].first + clusters_info[finished_clusters-1].second);
        //         // exit(1);
        //     }
        //     for(int i = 1; i < finished_clusters-1; ++i) {
        //         // if(clusters_info[i-1].first + clusters_info[i-1].second < clusters_info[i].first) {
        //         //     printf("ERROR: GAP detected between cluster %d (starting at %d and ending at %d size %d) and cluster %d (starting at %d)\n", i-1, clusters_info[i-1].first, clusters_info[i-1].first + clusters_info[i-1].second, clusters_info[i-1].second, i, clusters_info[i].first);
        //         //     // exit(1);
        //         // }
        //         if(clusters_info[i].first + clusters_info[i].second > clusters_info[i+1].first) {
        //             printf("ERROR: overlapping clusters detected between cluster %d (starting at %d and size %d) and cluster %d (starting at %d and size %d)\n", i, clusters_info[i].first, clusters_info[i].second, i+1, clusters_info[i+1].first, clusters_info[i+1].second);
        //             // exit(1);
        //         }
        //     }
        //     // exit(1);
        //     //check if indexes are correctly 
        //     //sort indexes using thrust
        //     std::vector<bool> index_used(n_points, false);
        //     int max_idx = 0;
        //     for(int i = 0; i < finished_clusters; ++i){
        //         int start = final_clusters_starts[i];
        //         int size = final_clusters_sizes[i];
        //         thrust::device_vector<int> d_indexes(gpu_nodes_buckets.ptr() + start, gpu_nodes_buckets.ptr() + start + size);
        //         // thrust::sort(d_indexes.begin(), d_indexes.end());
        //         //copy back to host
        //         std::vector<int> h_indexes(size);
        //         cudaMemcpy(
        //             h_indexes.data(),
        //             thrust::raw_pointer_cast(d_indexes.data()),
        //             size * sizeof(int),
        //             cudaMemcpyDeviceToHost
        //         );
        //         //check if indexes are sequential
        //         for(int i = 0; i < size; ++i) {
        //             // printf("Cluster %d, index %d: %d\n", i, start + i, h_indexes[i]);
        //             index_used[h_indexes[i]] = true;
        //             if(h_indexes[i] > max_idx) {
        //                 max_idx = h_indexes[i];
        //             }
        //         }
        //         // if(size < 10){
        //         //     printf("Cluster: start %d, size %d\n", start, size);
        //         //     //print indexes
        //         //     printf("Indexes: ");
        //         //     for(int j = 0; j < size; ++j) {
        //         //         printf("%d ", h_indexes[j]);
        //         //     }
        //         //     printf("\n");
        //         // }
        //     }
        // //     //check if all indexes are used
        //     for(int i = 0; i < max_idx; ++i) {
        //         if(!index_used[i]) {
        //             printf("ERROR: index %d is not used in any cluster\n", i);
        //             // exit(1);
        //         }
        //     }
        // }
        // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        return build_treeinfo(
            final_clusters_starts,
            final_clusters_sizes,
            finished_clusters,
            not_finished_clusters,
            gpu_nodes_buckets.ptr(),
            max_bucket_size
        );
    }

    // void destroy() {
    //     if(!initialized) {
    //         printf("WARNING: HierarchicalKMeans not initialized.\n");
    //         return;
    //     }
    //     for(auto& stream : streams) {
    //         stream.~KMeansStream();
    //     }
    //     indexes.free();
    //     initialized = false;
    // }

};
#endif // RECURSIVE_KMEANS_CU