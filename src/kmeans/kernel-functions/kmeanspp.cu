__global__
void initialize_first_cent_kmeanspp(float* dataset, uint dataset_size, 
        uint dim,
        float* centroids, uint* chosen_centroids){
    
    __shared__ uint r;

    //Initialize the centroids with the first K points

    if(threadIdx.x == 0){
        curandStateMRG32k3a_t  state;
        curand_init(blockIdx.x,
            0,
            0,
            &state);
        
        r = curand(&state) % dataset_size;
        chosen_centroids[0] = r;
    }
    __syncthreads();
    
    for(int i = threadIdx.x; i < dim; i+=blockDim.x){
        // printf("bid = %i, c[%i] = %f\n",r,threadIdx.x,dataset[i+r*dim]);
        centroids[i] = dataset[i+r*dim];
    }
}

// For each point in the dataset, compute its distance to the newly added centroid,
// and update its label, upper bound, and lower bound accordingly
// Also, find the point with the maximum distance to its nearest centroid
// to be used as the next centroid
// TODO: it is possible to optimize the search for the next centroid using atomics
__global__
void find_new_centroid_kmeanspp(float* dataset, uint dataset_size, 
        float* centroids, 
        uint k, // current number of centroids 
        uint dim, 
        uint* labels,
        float* upperbounds, float* lowerbounds,
        uint* chosen_centroids,
        uint* candidates, float* max_min_cent_dist
){

    // initialize variables
    uint warpIdx = threadIdx.x / WARP_SIZE;
    uint laneIdx = threadIdx.x % WARP_SIZE;
    uint cent = k-1; // index of the the current centroid added

    uint candidate_to_centroid; // point that has the farthest nearest centroid
    float max_dist = 0.0; // distance to farthest nearest centroid
    int nwarps = blockDim.x / WARP_SIZE;

    for(uint i = warpIdx+blockIdx.x*nwarps; i < dataset_size; i += nwarps*blockDim.x){
        ////////////////////////
        // CALCULATE DISTANCE //
        ////////////////////////
        float4 a,b;
        float s = 0.0f;
        uint nf = dim/4;
        for(uint d=laneIdx; d < nf; d+=WARP_SIZE){
            a = reinterpret_cast<float4*>(dataset)[i*nf+d];
            b = reinterpret_cast<float4*>(centroids)[cent*nf+d];
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
        s += __shfl_xor_sync( 0xffffffff, s,  1); // assuming warpSize=32
        s += __shfl_xor_sync( 0xffffffff, s,  2); // assuming warpSize=32
        s += __shfl_xor_sync( 0xffffffff, s,  4); // assuming warpSize=32
        s += __shfl_xor_sync( 0xffffffff, s,  8); // assuming warpSize=32
        s += __shfl_xor_sync( 0xffffffff, s, 16); // assuming warpSize=32	
        float new_dist = s;
        ////////////////////////
        ////////////////////////

        // upperbounds[i] == min dist
        // lowerbounds[i] == sec min dist
        if(new_dist < upperbounds[i]){
            lowerbounds[i] = upperbounds[i];
            upperbounds[i] = new_dist;
            labels[i] = cent;
        }
        else{
            if(new_dist < lowerbounds[i]){
                lowerbounds[i] = new_dist;
            }
            new_dist = upperbounds[i];
        }
        if(new_dist > max_dist){
            // uint not_chosen = 1;
            // for(uint l=laneIdx; l < k; l+=WARP_SIZE){
            //     if(chosen_centroids[l] == i){
            //         not_chosen = 0;
            //     }
            // }
            // if(__all_sync(FULL, not_chosen)){
                max_dist = new_dist;
                candidate_to_centroid=i; //i is candidate to next centroid
            // }
        }

    }
    if(laneIdx == 0){
        // printf("%u %f\n",candidate_to_centroid,max_dist);
        max_min_cent_dist[blockIdx.x*nwarps+warpIdx] = max_dist;
        candidates[blockIdx.x*nwarps+warpIdx] = candidate_to_centroid;
    }
       
}

// Find the point with the maximum distance to its nearest centroid
// and append it to the centroids list
// (following the KMeans++ initialization method (Arthur and Vassilvitskii, 2007))
// This kernel is launched with a single block
__global__
void append_centroid_kmeanspp(float* dataset, uint dataset_size, 
        float* centroids, uint k, uint dim,
        uint* chosen_centroids,
        uint* candidates, float* max_min_cent_dist, int total_candidates
){
    __shared__ float sh_max_dist[MAX_THREADS];
    __shared__ uint sh_next_cent[MAX_THREADS];
    
    float max = 0.0;
    uint next_cent=dataset_size+1;
    for(int i = threadIdx.x; i < total_candidates; i+=blockDim.x){
        if(max < max_min_cent_dist[i]){
            max = max_min_cent_dist[i];
            next_cent = candidates[i];
        }
    }
    sh_max_dist[threadIdx.x]=max;
    sh_next_cent[threadIdx.x]=next_cent;
    __syncthreads();

    for (uint s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            if(sh_max_dist[threadIdx.x] < sh_max_dist[threadIdx.x + s]){
                sh_max_dist[threadIdx.x] = sh_max_dist[threadIdx.x + s];
                sh_next_cent[threadIdx.x] = sh_next_cent[threadIdx.x + s];

            }
        }
        __syncthreads();
    }
    uint new_cent = sh_next_cent[0];
    if(threadIdx.x == 0){
        // if(new_cent >= dataset_size){
        //     printf("ERROR: new cent == %u dist == %f\n",sh_next_cent[0],sh_max_dist[0]);
        // }
        chosen_centroids[k] = new_cent;
    }
    for(int i = threadIdx.x; i < dim; i+=blockDim.x){
        centroids[i+k*dim] = dataset[i+new_cent*dim];
    }
}


__global__
void label_last_centroid_kmeanspp(float* dataset, uint dataset_size, 
        float* centroids, uint k, uint dim, 
        uint* labels,
        float* upperbounds, float* lowerbounds
){

    // initialize variables
    uint warpIdx = threadIdx.x / WARP_SIZE;
    uint laneIdx = threadIdx.x % WARP_SIZE;
    uint cent = k-1;

    for(uint i = warpIdx+blockIdx.x*N_WARPS; i < dataset_size; i += N_WARPS*N_BLOCKS){
        ////////////////////////
        // CALCULATE DISTANCE //
        ////////////////////////
        float4 a,b;
        float s = 0.0f;
        uint nf = dim/4;
        for(uint d=laneIdx; d < nf; d+=WARP_SIZE){
            a = reinterpret_cast<float4*>(dataset)[i*nf+d];
            b = reinterpret_cast<float4*>(centroids)[cent*nf+d];
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
        s += __shfl_xor_sync( 0xffffffff, s,  1); // assuming warpSize=32
        s += __shfl_xor_sync( 0xffffffff, s,  2); // assuming warpSize=32
        s += __shfl_xor_sync( 0xffffffff, s,  4); // assuming warpSize=32
        s += __shfl_xor_sync( 0xffffffff, s,  8); // assuming warpSize=32
        s += __shfl_xor_sync( 0xffffffff, s, 16); // assuming warpSize=32	
        float new_dist = s;
        ////////////////////////
        ////////////////////////

        // upperbounds[i] == min dist
        // lowerbounds[i] == sec min dist
        if(new_dist < lowerbounds[i]){
            if(new_dist < upperbounds[i]){
                lowerbounds[i] = upperbounds[i];
                upperbounds[i] = new_dist;
                labels[i] = cent;
            }
            else{
                lowerbounds[i] = new_dist;
            }
        }
    }
}

__global__
void sum_all_points_to_centroid(
        float* dataset, uint dataset_size, 
        float* centroids, uint k, uint dim, 
        uint* labels,
        int* label_count,
        float* new_centroids
    ){
    uint tid = threadIdx.x + blockDim.x*blockIdx.x;
    if(tid < dataset_size*dim){
        uint p = tid / dim;
        uint d = tid % dim;
        uint c = labels[p];
        if(d == 0){
            atomicAdd(&label_count[c],1);
        }
        atomicAdd(&new_centroids[c*dim+d],dataset[tid]);
    }
}

__global__
void divide_sum_by_count(
        float* new_centroids, uint k, uint dim,
        int* label_count,
        uint* old_label_count
    ){
    uint tid = threadIdx.x + blockDim.x*blockIdx.x;
    if(tid < k*dim){
        uint c = tid / dim;
        uint d = label_count[c];
        new_centroids[tid]=new_centroids[tid]/d;

        //save the value to the next iteration
        if(tid % dim == 0)
            old_label_count[c]=d; 
    }
}

__global__
void calculate_centroid_shift(
        float* centroids, 
        float* new_centroids, 
        uint dim, uint k, 
        float* centroid_shift, float* max_centroid_shift
    ){
    __shared__ float sm_centshift;
    //////////////////////////////////
    //   calculate centroid shift   //
    //////////////////////////////////
    float4 a,b;
    float s = 0.0f;
    uint nf = dim/4;
    for(uint t=threadIdx.x; t < nf; t+=blockDim.x){
        a = reinterpret_cast<float4*>(new_centroids)[blockIdx.x*nf+t];
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
        centroid_shift[blockIdx.x]=sm_centshift;
        atomicMaxFloat(max_centroid_shift,sm_centshift);
    }
}



__global__
void update_global_bounds(
        uint* labels, uint dataset_size,
        float* lowerbounds, float* upperbounds, 
        float* centroid_shift, float* gm_max_centroid_shift
    ){
    float max_centroid_shift = *gm_max_centroid_shift;

    for(uint i=threadIdx.x+blockDim.x*blockIdx.x;
            i<dataset_size;
            i+=blockDim.x*gridDim.x){
        lowerbounds[i]-=max_centroid_shift;
        uint nearest = labels[i];
        upperbounds[i]+=centroid_shift[nearest];
    }

}
    // __shared__ float sh_sqrdNormError;
    // float local_sqrdNormError;
    // if(threadIdx.x == 0)
    //     sh_sqrdNormError = 0;
    
    // __syncthreads();

    // if( tid < k*dim){
    //     local_sqrdNormError = centroids[tid] - new_centroids[tid];
    //     local_sqrdNormError = local_sqrdNormError*local_sqrdNormError;
    //     atomicAdd(&sh_sqrdNormError,local_sqrdNormError);
    // }

    // __syncthreads();

    // if(threadIdx.x == 0)
    //     atomicAdd(sqrdNormError,local_sqrdNormError);