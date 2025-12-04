#ifndef __ALIGN_MEMORY_CU__
#define __ALIGN_MEMORY_CU__

__global__
void align_memory(float* in, float* out, uint n_points, uint dim, uint logic_dim){
    uint tid = blockIdx.x*blockDim.x+threadIdx.x;
    if(tid < n_points){
        for(uint i = 0; i < dim; i++){
            out[tid*logic_dim+i] = in[tid*dim+i];
        }
        for(uint i = dim; i < logic_dim; i++){
            out[tid*logic_dim+i] = 0.0;
        }
    }
}

__global__
void align_memory_persistent(float* in, float* out, uint n_points, uint dim, uint logic_dim){
    uint tid = blockIdx.x*blockDim.x+threadIdx.x;
    for(int i = tid; i < n_points; i += gridDim.x*blockDim.x){
    // if(tid < n_points){
        for(uint j = 0; j < dim; j++){
            out[i*logic_dim+j] = in[i*dim+j];
        }
        for(uint j = dim; j < logic_dim; j++){
            out[i*logic_dim+j] = 0.0;
        }
    }
}

__global__
void unalign_memory(float* in, float* out, uint n_points, uint dim, uint logic_dim){
    uint tid = blockIdx.x*blockDim.x+threadIdx.x;
    if(tid < n_points){
        for(uint i = 0; i < dim; i++){
            out[tid*dim+i]=in[tid*logic_dim+i];
        }
    }
}


#endif // __ALIGN_MEMORY_CU__