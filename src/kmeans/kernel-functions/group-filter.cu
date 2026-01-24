#ifndef GROUP_FILTER_CU
#define GROUP_FILTER_CU

#include "gpu-utils.cu"

__global__
void organize_group_filter( 
		uint* group_filter_labels,
        uint k,        
        uint t_groups,
        uint* group_filter_location, 
        uint* group_filter_cents
    ){
    __shared__ uint group_size;

    if(threadIdx.x == 0){
        group_filter_location[0] = 0;
    }
    for(int i = 0; i < t_groups; i++){
        if(threadIdx.x == 0){
            group_size = 0;
        }
        __syncthreads();

        for(uint j = threadIdx.x; j < k; j+=blockDim.x){
            if(group_filter_labels[j] == i){
                uint pos = atomicAdd(&group_size,1) + group_filter_location[i]; 
                group_filter_cents[pos] = j;
            }
        }
        __syncthreads();
        if(threadIdx.x == 0){
            group_filter_location[i+1] = group_filter_location[i] + group_size;
        }
    }
}

/////////////////////////
//// ICPADS24 VERSION ///
/////////////////////////

//////////////////////////
/// !!!! OUTDATED !!!! ///
//////////////////////////

__global__
void group_filter_assignment(float* centroids, uint k, 
        uint dim, uint t_groups, 
        float* group_centroid, uint* labels){

    // extern __shared__ uint sm[];
    // float* group_centroid = sm; //usage: t_groups*dim

    // initialize variables
    uint warpIdx = threadIdx.x / WARP_SIZE;
    uint laneIdx = threadIdx.x % WARP_SIZE;
    int nwarps = blockDim.x / WARP_SIZE;

    for(uint i = warpIdx+blockIdx.x*nwarps; i < k; i += nwarps*blockDim.x){
        float min_dist = MAX_FLOAT;
        uint nearest_cent=0;
        for(uint j = 0; j < t_groups; j++){
            ////////////////////////
            // CALCULATE DISTANCE //
            ////////////////////////
    		// float4 a,b;
    		// float s = 0.0f;
            // uint nf = dim/4;
	    	// for(uint d=laneIdx; d < nf; d+=WARP_SIZE){
            //    a = reinterpret_cast<float4*>(centroids)[i*nf+d];
            //     b = reinterpret_cast<float4*>(group_centroid)[j*nf+d];
            //     float4 diff;
            //     diff.x = a.x - b.x;
            //     diff.y = a.y - b.y;
            //     diff.z = a.z - b.z;
            //     diff.w = a.w - b.w;
            //     s+=diff.x*diff.x;
            //     s+=diff.y*diff.y;
            //     s+=diff.z*diff.z;
            //     s+=diff.w*diff.w;
            // }
            // s += __shfl_xor_sync( 0xffffffff, s,  1); // assuming warpSize=32
            // s += __shfl_xor_sync( 0xffffffff, s,  2); // assuming warpSize=32
            // s += __shfl_xor_sync( 0xffffffff, s,  4); // assuming warpSize=32
            // s += __shfl_xor_sync( 0xffffffff, s,  8); // assuming warpSize=32
            // s += __shfl_xor_sync( 0xffffffff, s, 16); // assuming warpSize=32	
            // float new_dist = s;
            float new_dist = warp_euclidean_distance_float4(
                &centroids[i*dim],
                &group_centroid[j*dim],
                dim,
                laneIdx
            );
            ////////////////////////
            ////////////////////////

            if(new_dist < min_dist){
                min_dist = new_dist;
                nearest_cent=j;
            }

        }
        if(laneIdx == 0){
            labels[i] = nearest_cent;
        }
        
    }
}

__global__
void group_filter_update(float* centroids, uint k, 
        uint dim,
        float* group_centroid, uint* labels){

    __shared__ uint count;
    extern __shared__ uint sm[];
    float* new_centroid = (float*)sm; //usage: dim

    // initialize variables
    uint warpIdx = threadIdx.x / WARP_SIZE;
    uint laneIdx = threadIdx.x % WARP_SIZE;

    for(int j = threadIdx.x; j < dim; j+= blockDim.x){
        new_centroid[j] = 0;
    }
    if(threadIdx.x == 0){
        count = 0;
    }
    __syncthreads();

    for(uint i = warpIdx; i < k; i += N_WARPS){
        if(labels[i] == blockIdx.x){
            for(int j = laneIdx; j < dim; j+= WARP_SIZE){
                atomicAdd(&new_centroid[j], centroids[i*dim+j]);
            }
            if(laneIdx == 0){
                atomicAdd(&count,1);
            }
        }
    }
    __syncthreads();
    
    for(int j = laneIdx; j < dim; j+= WARP_SIZE){
        group_centroid[blockIdx.x*dim+j] = new_centroid[j]/(float)count;
    }

    #if DEBUG_GROUP_FILTER
    if(threadIdx.x == 0 && blockIdx.x == 0){
        for(int i = 0; i < k; i+= 1){
            printf("%d ",labels[i]);
        }
        printf("\n");
    }
    #endif
}

__global__
void organize_group_filter( 
        float* centroids, uint k, uint dim, 
        uint t_groups,
		uint* group_filter_labels, 
        uint* group_filter_cents, uint* group_filter_location,
        uint* groups_with_only_one_element
        // ,uint* sm
    ){
    extern __shared__ uint sm[];
    uint* group_size = (uint*)sm; //usage: t_group
    uint* group_count = (uint*)&sm[t_groups]; //usage: t_group
    uint* invalid_group = (uint*)&sm[2*t_groups]; //usage: t_group

    uint* group_one_element_count = (uint*)&sm[3*t_groups]; //usage: 1 uint

    for(uint i=threadIdx.x; i<t_groups; i+=blockDim.x){
        group_size[i]=0;
        group_count[i]=0;
        invalid_group[i] = 0;
    }
    *group_one_element_count = 0;
    __syncthreads();

    for(uint i=threadIdx.x; i<k; i+=blockDim.x){
        uint g = group_filter_labels[i];
        if(g > t_groups)
            printf("ERROR in %s %d: invalid group index (g[%d] = %d > %d)\n",__FILE__, __LINE__,i,g,t_groups);
        atomicAdd(&group_size[g],1);
    }
    __syncthreads();
   
    for(uint i=threadIdx.x; i<t_groups; i+=blockDim.x){
        if(group_size[i] < 1){
            invalid_group[i] = 1;
            atomicAdd(group_one_element_count,1);
        }
    }
    __syncthreads();

    if(threadIdx.x == 0)
        *groups_with_only_one_element=*group_one_element_count;
    if(*group_one_element_count){
        // if(threadIdx.x == 0)
        //         printf("ERROR in %s %d: invalid groups = %d)\n",__FILE__, __LINE__,*group_one_element_count);
 
        // if(threadIdx.x == 0){
        //     for(uint i=0; i<t_groups; i++)
        //         printf("size[%u] %u\n",i,group_size[i]);
        //     printf("label: ");
        //     for(uint i=0; i<k; i++)
        //         printf("%u ",group_filter_labels[i]);
        //     printf("\n");
        // }
        uint head = 0;
        uint tail = t_groups;
        for(uint i=0; i < *group_one_element_count; i++ ){
            while(head < t_groups && !invalid_group[head]){
                head++;
            }
            while(tail > 0 && invalid_group[tail]){
                tail--;
            }
            // if(threadIdx.x == 0)
            //     printf("head %u tail %u size[%u] %u size[%u] %u\n",head,tail,head,group_size[head],tail,group_size[tail]);
            if(tail > head){
                for(uint d=threadIdx.x; d<dim; d+=blockDim.x){
                    centroids[head*dim+d] = centroids[tail*dim+d];
                }
                head++;
                tail--;
            }
        }
        return;
    }

    //PREFIX SUM
    if(threadIdx.x == 0){
        group_filter_location[0]=0;
        uint sum = 0;
        for(uint i=1; i<t_groups+1; i++){
            sum+=group_size[i-1];
            group_filter_location[i]=sum;
        }
    }
    __syncthreads();

    for(uint i=threadIdx.x; i<k; i+=blockDim.x){
        uint g = group_filter_labels[i];
        uint begin = group_filter_location[g];
        uint shift = atomicAdd(&group_count[g],1);
        group_filter_cents[begin+shift] = i;
    }

    // if(threadIdx.x == 0){
    //     // for(uint i=0; i<t_groups; i++)
    //     //     printf("size[%u] %u\n",i,group_size[i]);
    //     printf("label: ");
    //     for(uint i=0; i<k; i++)
    //         printf("%u ",group_filter_labels[i]);
    //     printf("\n");
    //     printf("group: ");
    //     for(uint i=0; i<k; i++)
    //         printf("%u ",group_filter_cents[i]);
    //     printf("\n");
    // }
}

__global__
void organize_cents_in_memory(float* in, float* out, uint n_cents, uint dim,
		uint* group_filter_labels, 
        uint* group_filter_cents, uint* group_filter_location){

    uint g = 0;
    for(uint i = blockIdx.x*blockDim.x+threadIdx.x; i < n_cents*dim; i+=gridDim.x*blockDim.x){
        uint pos = i/dim;
        uint d = i % dim;
        uint cent = group_filter_cents[pos];
        while(pos > group_filter_location[g+1])
            g++;
        if(d == 0){
            group_filter_labels[pos] = g;
            group_filter_cents[pos] = pos;
        }
        out[pos*dim+d] = in[cent*dim+d];
    }
}

#endif // GROUP_FILTER_CU