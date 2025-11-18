#include <curand_kernel.h>

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

__global__
void setMaxFloat(float* mem, uint size){
    uint i = blockIdx.x*blockDim.x+threadIdx.x;
    if(i < size)
        mem[i]=MAX_FLOAT;
}