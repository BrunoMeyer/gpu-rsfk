#ifndef KMEANSGPU_H

#define KMEANSGPU_H

#include "gpu_ptr.h"

class KMeansInfo {
private:

public:
    GpuPtr<float> points;      // dataset N × D
    GpuPtr<uint> labels;     // N
    GpuPtr<float> centroids;   // K × D
    GpuPtr<float> dist_to_centroids;   // N

    uint n_points = 0;
    uint dim = 0;
    uint logic_dim = 0;
    uint n_clusters = 0;

    KMeansInfo(float* raw_points, uint n_points, uint dim, uint n_clusters)
        : n_points(n_points),
        dim(dim),
        logic_dim((dim + 3) / 4 * 4),   // align to 4
        n_clusters(n_clusters)
    {
        if(logic_dim != dim) {
            points.alloc(n_points * logic_dim);
        } else {
            points.attach(raw_points, n_points * dim);
        }
        labels.alloc(n_points);
        centroids.alloc(n_clusters * logic_dim);
        dist_to_centroids.alloc(n_points);
    }

    // disable copy constructor and copy assignment
    KMeansInfo(const KMeansInfo&) = delete;
    KMeansInfo& operator=(const KMeansInfo&) = delete;

    // move constructor
    KMeansInfo(KMeansInfo&& other) noexcept
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
    KMeansInfo& operator=(KMeansInfo&& other) noexcept {
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
    ~KMeansInfo() = default;
};

// void kmeansGpu(float* d_dataset, uint dataset_size, 
//     uint dim, uint k, 
// 	uint max_it, 
// 	uint check_method, float tolerance, 
// 	uint initialization_method, //0 - random, 1 - from cpu
// 	uint t_groups,
// 	uint verbosity,
//     uint* d_labels, float* d_centroids // <-- outputs
// );

#include "./kmeans.cu"

#endif