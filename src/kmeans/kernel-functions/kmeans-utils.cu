#ifndef KMEANS_UTILS_CU
#define KMEANS_UTILS_CU
template <typename T>
__global__
void count_labels_kernel(T* labels, int* label_counts, int num_points, int num_clusters) {
    extern __shared__ int shared_counts[];
    for (int i = threadIdx.x; i < num_clusters; i += blockDim.x) {
        shared_counts[i] = 0;
    }
    __syncthreads();

    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_points; i += blockDim.x * gridDim.x) {
        int label = labels[i];
        if (label >= 0 && label < num_clusters) {
            atomicAdd(&shared_counts[label], 1);
        }
    }
    __syncthreads();

    for (int i = threadIdx.x; i < num_clusters; i += blockDim.x) {
        atomicAdd(&label_counts[i], shared_counts[i]);
    }
}
#endif