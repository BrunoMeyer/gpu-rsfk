#ifndef KMEANS_INITIALIZE_CU
#define KMEANS_INITIALIZE_CU

#include <curand_kernel.h>

template <bool INDIRECT_POINTS=false>
__global__
void initialize_with_given_cent(
		float* dataset, uint dataset_size, uint dim,
		float* centroids, uint first_centroid_idx, 
        int* indexes = nullptr
){
    //debug: print all parameters
    // if(threadIdx.x == 0 && blockIdx.x == 0){
    //     printf("pointer dataset: %p\n", dataset);
    //     //access first point
    //     printf("first point: ");
    //     for(int i = 0; i < dim; i++){
    //         printf("%f ", dataset[i]);
    //     }
    //     printf("pointer centroids: %p\n", centroids);
    //     //access first centroid
    //     printf("first centroid: ");
    //     for(int i = 0; i < dim; i++){
    //         printf("%f ", centroids[i]);
    //     }
    //     printf("pointer indexes: %p\n", indexes);
    //     //access first 10 indexes
    //     printf("first 10 indexes: ");
    //     for(int i = 0; i < min(10u, dataset_size); i++){
    //         printf("%d ", indexes[i]);
    //     }
    //     printf("dataset_size: %u, dim: %u, first_centroid_idx: %u\n", dataset_size, dim, first_centroid_idx);
    // }
    // return;

    int idx;
    if constexpr (INDIRECT_POINTS){
        idx = indexes[first_centroid_idx];
        if(idx == -1){
            __shared__ int first_valid_idx;
            if(threadIdx.x == 0){
                first_valid_idx = -1;
            }
            __syncthreads();
            for(int i = threadIdx.x; i < dataset_size; i+=blockDim.x){
                int temp_idx = indexes[(i+first_centroid_idx)%dataset_size];
                if(temp_idx != -1){
                    atomicMin(&first_valid_idx, temp_idx);
                    break;
                }
            }
            __syncthreads();
            idx = first_valid_idx;
        }
    } else{
        idx = first_centroid_idx;
    }
	for(uint i = threadIdx.x + blockIdx.x*blockDim.x; i < dim; i+=blockDim.x*gridDim.x){
		centroids[i] = dataset[idx*dim + i];
	}
}

__global__
void initialize(float* dataset, uint dataset_size, 
        uint dim,
        float* centroids){
    
    __shared__ uint r;

    //Initialize the centroids with the first K points

    if(threadIdx.x == 0){
        curandStateMRG32k3a_t  state;
        curand_init(blockIdx.x,
            0,
            0,
            &state);
        
        r = curand(&state) % dataset_size;
        // r=blockIdx.x;
    }
    __syncthreads();
    
    for(int i = threadIdx.x; i < dim; i+=blockDim.x){
        // printf("bid = %i, c[%i] = %f\n",r,threadIdx.x,dataset[i+r*dim]);
        centroids[i+dim*blockIdx.x] = dataset[i+r*dim];
    }
}
        // printf("bid = %i, r = %u, dim=%u\n",blockIdx.x,r,dim);

#endif // KMEANS_INITIALIZE_CU