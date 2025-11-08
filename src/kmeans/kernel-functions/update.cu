__global__
void update(float* centroids,  
        uint dim, uint k, 
        uint* labels, int* label_partial_count,
        uint* global_old_label_count,
        float* global_new_centroids, float* global_partial_centroids,
        float* centroid_shift, float* max_centroid_shift,  
		uint* group_filter,
        float* max_group_shift,
        uint* reassignments,
        uint use_shared_memory
    ){
    
    __shared__ float sm_centshift;
    __shared__ int label_count;
    extern __shared__ uint sm[];
    float* centroid_diff;
    if(use_shared_memory){

        centroid_diff = (float*)sm;
        for(uint j = threadIdx.x; j < dim; j += blockDim.x){
            centroid_diff[j] = global_partial_centroids[blockIdx.x*dim+j];
        }
    }
    else{
        centroid_diff = &global_partial_centroids[blockIdx.x*dim];
    }

    if(threadIdx.x == 0){
        label_count = 0;
        sm_centshift = 0;
    }
    __syncthreads();

    for(uint i = k*threadIdx.x+blockIdx.x; i < N_BLOCKS*k; i+=k*blockDim.x){
        atomicAdd(&label_count,label_partial_count[i]);
    }
    __syncthreads();
    


    uint old_count = global_old_label_count[blockIdx.x];
    for(uint i = blockIdx.x*dim + dim*k; i < k*dim*N_BLOCKS*N_WARPS; i += dim*k){
        for(uint j = threadIdx.x; j < dim; j += blockDim.x){
            centroid_diff[j] += global_partial_centroids[i+j];
        }
    }
    // if(label_count == 0){
    //     if(threadIdx.x == 0)
    //         printf("whats going on??? %d -> %d\n",old_count,label_count);
    // }

    for(uint i = threadIdx.x; i < dim; i += blockDim.x){
        global_new_centroids[i + blockIdx.x*dim] = (centroids[i + blockIdx.x*dim]*old_count+centroid_diff[i])/label_count;
    }
    __syncthreads();

    //////////////////////////////////
    //   calculate centroid shift   //
    //////////////////////////////////
    float4 a,b;
    float s = 0.0f;
    uint nf = dim/4;
    for(uint t=threadIdx.x; t < nf; t+=blockDim.x){
        a = reinterpret_cast<float4*>(global_new_centroids)[blockIdx.x*nf+t];
        b = reinterpret_cast<float4*>(centroids)[blockIdx.x*nf+t];
        float4 diff;
        diff.x = a.x - b.x;
        diff.y = a.y - b.y;
        diff.z = a.z - b.z;
        diff.w = a.w - b.w;
        s+=diff.x*diff.x;
        s+=diff.y*diff.y;
        s+=diff.z*diff.z;
        s+=diff.w*diff.w;
    }
    atomicAdd(&sm_centshift,s);
    __syncthreads();

    if(threadIdx.x == 0){
        global_old_label_count[blockIdx.x]=label_count;
        centroid_shift[blockIdx.x]=sm_centshift;
        atomicMaxFloat(max_centroid_shift,sm_centshift);

        uint g = group_filter[blockIdx.x];
        atomicMaxFloat(&max_group_shift[g],sm_centshift);

        uint r = abs((int)old_count-label_count);
        atomicAdd(reassignments,r);
    }
}


    // if(threadIdx.x == 0 && blockIdx.x == 0){
    //     printf("%f\n",centroid[warpIdx*dim]);
    // }