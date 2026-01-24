#ifndef KMEANSGPU_H

#define KMEANSGPU_H

#include "gpu_ptr.h"
#include "defines.h"
#include "kernel-functions/align-memory.cu"
#include "kernel-functions/kmeans-utils.cu"

class KMeansGPU {
private:
    bool need_align = true;
    GpuPtr<int> labels_counts;
    int max_shared_mem = 0;

    

public:
    int devUsed = 0;
    int max_threads = MAX_THREADS;
    int nthreads = 0;
    int nblocks = 0;

    GpuPtr<float> points;      // dataset N × D
    GpuPtr<uint> labels;     // N
    GpuPtr<float> centroids;   // K × D
    GpuPtr<float> dist_to_centroids;   // N

    uint n_points = 0;
    uint dim = 0;
    uint logic_dim = 0;
    uint n_clusters = 0;

    KMeansGPU(float* raw_points, uint n_points, uint dim, uint n_clusters)
        : n_points(n_points),
        dim(dim),
        logic_dim((dim + 3) / 4 * 4),   // align to 4
        n_clusters(n_clusters)
    {
        points.attach(raw_points, n_points * dim);
        if(logic_dim == dim) {
            need_align = false;
        } else {
            need_align = true;
        }
        labels.alloc(n_points);
        centroids.alloc(n_clusters * logic_dim);
        dist_to_centroids.alloc(n_points);
    }

    void init() {
        cudaSetDevice(devUsed);
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, devUsed);

        max_threads = deviceProp.maxThreadsPerBlock;
        if(max_threads > MAX_THREADS){
            printf("WA in %s %d: The macro MAX_THREADS (%d) is lower than the device max threads per block (%d). File \"define.h\" need to be updated \n", __FILE__,__LINE__, MAX_THREADS,max_threads);
            max_threads = MAX_THREADS;
        }
        nthreads = deviceProp.maxThreadsPerMultiProcessor / 2;
        nblocks = deviceProp.multiProcessorCount*(max_threads/nthreads);
        max_shared_mem = deviceProp.sharedMemPerBlock;

        if(need_align) {
            float* attached_ptr = points.ptr();
            points.alloc(n_points * logic_dim); // realloc should not be used, because points is attached to external memory
            align_memory_persistent<<<nblocks, nthreads>>>(attached_ptr, points.ptr(), n_points, dim, logic_dim);
            need_align = false;
        }
    }
    
    // int* countLabels() {
    void countLabels() {
        labels_counts.alloc(n_clusters);
        cudaMemset(labels_counts.ptr(), 0, n_clusters * sizeof(int));

        int shared_mem_size = nthreads * sizeof(int);
        if(shared_mem_size > max_shared_mem) {
            printf("Warning: shared memory size (%d) is larger than the device max shared memory per block (%d).", shared_mem_size, max_shared_mem);
        }

        count_labels_kernel<<<nblocks, nthreads, shared_mem_size>>>(labels.ptr(), labels_counts.ptr(), n_points, n_clusters);

        cudaDeviceSynchronize();
        // return labels_counts.ptr();
    }

    // disable copy constructor and copy assignment
    KMeansGPU(const KMeansGPU&) = delete;
    KMeansGPU& operator=(const KMeansGPU&) = delete;

    // move constructor
    KMeansGPU(KMeansGPU&& other) noexcept
        : points(std::move(other.points)),
        labels(std::move(other.labels)),
        centroids(std::move(other.centroids)),
        dist_to_centroids(std::move(other.dist_to_centroids)),
        n_points(other.n_points),
        dim(other.dim),
        logic_dim(other.logic_dim),
        n_clusters(other.n_clusters)
    {
        other.n_points = 0;
        other.dim = 0;
        other.logic_dim = 0;
        other.n_clusters = 0;
    }

    // move assignment
    KMeansGPU& operator=(KMeansGPU&& other) noexcept {
        if (this != &other) {
            points = std::move(other.points);
            labels = std::move(other.labels);
            centroids = std::move(other.centroids);
            dist_to_centroids = std::move(other.dist_to_centroids);

            n_points = other.n_points;
            dim = other.dim;
            logic_dim = other.logic_dim;
            n_clusters = other.n_clusters;

            other.n_points = 0;
            other.dim = 0;
            other.logic_dim = 0;
            other.n_clusters = 0;
        }
        return *this;
    }

    // destructor (default is enough)
    ~KMeansGPU() = default;

};

using KMeansInfo = KMeansGPU;

#include "./kmeans.cu"

#endif