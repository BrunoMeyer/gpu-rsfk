__global__
void check_convergence(float* centroids, float* new_centroids, 
                        uint dim, uint k, float* sqrdNormError){

    // __shared__ float sh_sqrdNormError;
    // float local_sqrdNormError;
    // if(threadIdx.x == 0)
    //     sh_sqrdNormError = 0;
    
    // __syncthreads();

    // uint tid = threadIdx.x + blockDim.x*blockIdx.x;
    // if( tid < k*dim){
    //     local_sqrdNormError = centroids[tid] - new_centroids[tid];
    //     local_sqrdNormError = local_sqrdNormError*local_sqrdNormError;
    //     atomicAdd(&sh_sqrdNormError,local_sqrdNormError);
    // }

    // __syncthreads();

    // if(threadIdx.x == 0)
    //     atomicAdd(sqrdNormError,local_sqrdNormError);

    __shared__ float sh_sqrdNormError;
    if(threadIdx.x == 0)
        sh_sqrdNormError = 0;
    
    __syncthreads();

    for(uint i = threadIdx.x+blockIdx.x*dim; i < dim; i+=blockDim.x){
        float local_sqrdNormError;
        local_sqrdNormError = centroids[i] - new_centroids[i];
        local_sqrdNormError = local_sqrdNormError*local_sqrdNormError;
        atomicAdd(&sh_sqrdNormError,local_sqrdNormError);
    }

    __syncthreads();

    if(threadIdx.x == 0){
        atomicAdd(sqrdNormError,sh_sqrdNormError);
        // atomicMaxFloat(sqrdNormError,sh_sqrdNormError);
    }
}

__global__ void sumReduce(float* in, uint size, float* out){
    float sum = 0.0;
    for(int i = blockIdx.x*blockDim.x+threadIdx.x;i < size; i+=gridDim.x*blockDim.x){
        sum+=in[i];
    }
    atomicAdd(out,sum);
}