#ifndef KMEANS_UPDATE_CU
#define KMEANS_UPDATE_CU

__global__
void update_centroids(float* centroids,  
        uint dim, uint k, 
        int* label_change_count,
        uint* label_count,
        float* new_centroids, float* centroid_change
    ){

    int tid = blockDim.x*blockIdx.x + threadIdx.x;
    for(uint i = tid; i < dim*k; i += blockDim.x*gridDim.x){
        int c = i/dim;
        int new_count = label_change_count[c] + label_count[c];
        float change = centroid_change[i]/new_count;
        float old_value = centroids[i]*label_count[c]/new_count;
        new_centroids[i] = old_value + change;
    }

}

__global__
void update_label_count_and_centroid_shift(float* centroids,  
        uint dim, uint k, 
        int* label_change_count,
        uint* label_count,
        float* new_centroids,
        float* centroid_shift, float* max_centroid_shift,  
		uint* group_filter,
        float* max_group_shift,
        uint* reassignments){

    int tid = blockDim.x*blockIdx.x + threadIdx.x;
    __shared__ int reassignments_s;
    if(threadIdx.x == 0){
        reassignments_s = 0;
    }
    for(uint i = tid; i < k; i+=blockDim.x*gridDim.x){
        label_count[i]+=label_change_count[i];
        if(label_change_count[i] > 0)
            atomicAdd(&reassignments_s,label_change_count[i]);
    }
    __syncthreads();
    if(threadIdx.x == 0)
        atomicAdd(reassignments,reassignments_s);

    int nwarps = blockDim.x / warpSize;
    int wid = tid / nwarps;
    int nwa = blockDim.x*gridDim.x / nwarps;
    int lane = tid % nwarps;
    for(int i = wid; i < k; i += nwa){
        //////////////////////////////////
        //   calculate centroid shift   //
        //////////////////////////////////
        float shift = warp_euclidean_distance_float4(
            &new_centroids[i*dim],
            &centroids[i*dim],
            dim,
            lane
        );

        if(lane == 0){
            centroid_shift[i]=shift;
            atomicMaxFloat(max_centroid_shift,shift);

            uint g = group_filter[i];
            atomicMaxFloat(&max_group_shift[g],shift);
        }
    }
}


// ICPADS24
// __global__
// void update(float* centroids,  
//         uint dim, uint k, 
//         uint* labels, int* label_change_count,
//         uint* label_count,
//         float* new_centroids, float* centroid_change,
//         float* centroid_shift, float* max_centroid_shift,  
// 		uint* group_filter,
//         float* max_group_shift,
//         uint* reassignments,
//         uint use_shared_memory,
//         uint nblocks, uint nwarps
//     ){
    
//     __shared__ float sm_centshift;
//     __shared__ int dunno;
//     extern __shared__ uint sm[];
//     float* centroid_diff;
//     if(use_shared_memory){

//         centroid_diff = (float*)sm;
//         for(uint j = threadIdx.x; j < dim; j += blockDim.x){
//             centroid_diff[j] = centroid_change[blockIdx.x*dim+j];
//         }
//     }
//     else{
//         centroid_diff = &centroid_change[blockIdx.x*dim];
//     }

//     if(threadIdx.x == 0){
//         dunno = 0;
//         sm_centshift = 0;
//     }
//     __syncthreads();

//     for(uint i = k*threadIdx.x+blockIdx.x; i < nblocks*k; i+=k*blockDim.x){
//         atomicAdd(&dunno,label_change_count[i]);
//     }
//     __syncthreads();
    


//     uint old_count = label_count[blockIdx.x];
//     for(uint i = blockIdx.x*dim + dim*k; i < k*dim*nblocks*nwarps; i += dim*k){
//         for(uint j = threadIdx.x; j < dim; j += blockDim.x){
//             centroid_diff[j] += centroid_change[i+j];
//         }
//     }
//     // if(dunno == 0){
//     //     if(threadIdx.x == 0)
//     //         printf("whats going on??? %d -> %d\n",old_count,dunno);
//     // }

//     for(uint i = threadIdx.x; i < dim; i += blockDim.x){
//         new_centroids[i + blockIdx.x*dim] = (centroids[i + blockIdx.x*dim]*old_count+centroid_diff[i])/dunno;
//     }
//     __syncthreads();

//     //////////////////////////////////
//     //   calculate centroid shift   //
//     //////////////////////////////////
//     float4 a,b;
//     float s = 0.0f;
//     uint nf = dim/4;
//     for(uint t=threadIdx.x; t < nf; t+=blockDim.x){
//         a = reinterpret_cast<float4*>(new_centroids)[blockIdx.x*nf+t];
//         b = reinterpret_cast<float4*>(centroids)[blockIdx.x*nf+t];
//         float4 diff;
//         diff.x = a.x - b.x;
//         diff.y = a.y - b.y;
//         diff.z = a.z - b.z;
//         diff.w = a.w - b.w;
//         s+=diff.x*diff.x;
//         s+=diff.y*diff.y;
//         s+=diff.z*diff.z;
//         s+=diff.w*diff.w;
//     }
//     atomicAdd(&sm_centshift,s);
//     __syncthreads();

//     if(threadIdx.x == 0){
//         label_count[blockIdx.x]=dunno;
//         centroid_shift[blockIdx.x]=sm_centshift;
//         atomicMaxFloat(max_centroid_shift,sm_centshift);

//         uint g = group_filter[blockIdx.x];
//         atomicMaxFloat(&max_group_shift[g],sm_centshift);

//         uint r = abs((int)old_count-dunno);
//         atomicAdd(reassignments,r);
//     }
// }


//     // if(threadIdx.x == 0 && blockIdx.x == 0){
//     //     printf("%f\n",centroid[warpIdx*dim]);
//     // }

#endif // KMEANS_UPDATE_CU