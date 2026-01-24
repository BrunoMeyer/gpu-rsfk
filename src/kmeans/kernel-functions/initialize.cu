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
    uint idx;
    if constexpr (INDIRECT_POINTS){
        idx = indexes[first_centroid_idx];
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