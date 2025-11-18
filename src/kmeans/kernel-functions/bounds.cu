#include "gpu-utils.cu"

__global__
void init_bounds_and_assign_labels(float* dataset, uint dataset_size, 
        float* centroids, uint k, uint dim, 
        uint* labels, int* global_labels_count,
        float* upperbounds, float* lowerbounds,
        uint t_groups,
		uint* group_filter, float* group_lowerbounds,
        float* global_new_centroids
        ,uint* group_filter_cents, uint* group_filter_location
    ){


    // if(threadIdx.x == 0 && blockIdx.x == 0){
    //     printf("labels:");
    //     for(uint i = 0; i < k; i++){
    //         printf(" %u",group_filter[i]);
    //     }
    //     printf("\nlocation:");
    //     for(uint i = 0; i < t_groups+1; i++){
    //         printf(" %u",group_filter_location[i]);
    //     }
    //     printf("\ncents:");
    //     for(uint i = 0; i < k; i++){
    //         printf(" %u",group_filter_cents[i]);
    //     }
    // }


    // initialize variables
    uint warpIdx = threadIdx.x / WARP_SIZE;
    uint laneIdx = threadIdx.x % WARP_SIZE;

    int* labels_count = &global_labels_count[blockIdx.x*k];
    float* new_centroids = &global_new_centroids[blockIdx.x*dim*k*N_WARPS + warpIdx*dim*k];

    for(uint i = warpIdx+blockIdx.x*N_WARPS; i < dataset_size; i += N_WARPS*N_BLOCKS){
        for(uint j = laneIdx; j < t_groups; j+= WARP_SIZE){
            group_lowerbounds[t_groups*i+j] = MAX_FLOAT;
        }
        float min_dist = MAX_FLOAT;
        float secmin_dist = MAX_FLOAT;
        uint nearest_cent;
        for(uint j = 0; j < k; j++){
            ////////////////////////
            // CALCULATE DISTANCE //
            ////////////////////////
    		float4 a,b;
    		float s = 0.0f;
            uint nf = dim/4;
	    	for(uint d=laneIdx; d < nf; d+=WARP_SIZE){
               a = reinterpret_cast<float4*>(dataset)[i*nf+d];
                b = reinterpret_cast<float4*>(centroids)[j*nf+d];
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
            // float new_dist = euclidean_distance_sqr(&dataset[i],&centroids[j],dim,laneIdx);

            if(new_dist < secmin_dist){
                if(new_dist < min_dist){
                    secmin_dist = min_dist;
                    min_dist = new_dist;
                    nearest_cent=j;
                }
                else{
                    secmin_dist = new_dist;
                }
            }

            uint g = group_filter[j];
            if(new_dist < group_lowerbounds[t_groups*i+g])
                group_lowerbounds[t_groups*i+g]=new_dist;
        }
        if(laneIdx == 0){
            labels[i] = nearest_cent;
            upperbounds[i]=min_dist;
            lowerbounds[i]=secmin_dist;
            atomicAdd(&labels_count[nearest_cent],1);

            //The group that has the nearest centroid must be updated with the second minimum distance
            uint g = group_filter[nearest_cent];
            group_lowerbounds[t_groups*i+g]=secmin_dist;
        }
        // calculate the partial update for the new centroid
        ///////////////////////////
        // SUM POINT TO CENTROID //
        ///////////////////////////
		for(uint l=laneIdx; l < dim; l+=WARP_SIZE){
			// atomicAdd(&new_centroids[nearest_cent*dim+l], dataset[i*dim+l]);
			new_centroids[nearest_cent*dim+l] += dataset[i*dim+l];
		}


    }
}

__global__
void update_bounds(
        uint dataset_size, uint dim, uint k,
        uint* labels,
        float* lowerbounds, float* upperbounds, 
        float* centroid_shift, float* gm_max_centroid_shift,
        uint t_groups,
		float* group_lowerbounds,
        float* max_group_shift){
    float max_centroid_shift = *gm_max_centroid_shift;

    //updating GLOBAL FILTER
    for(uint i=threadIdx.x+blockDim.x*blockIdx.x;
            i<dataset_size;
            i+=blockDim.x*gridDim.x){
        lowerbounds[i]-=max_centroid_shift;
        uint nearest = labels[i];
        upperbounds[i]+=centroid_shift[nearest];
    }

    //updating GROUP FILTER
    for(uint i=threadIdx.x+blockDim.x*blockIdx.x;
            i<dataset_size*t_groups;
            i+=blockDim.x*gridDim.x){
        uint t = i%t_groups;
        // uint p = i/t_groups;
        group_lowerbounds[i]-=max_group_shift[t];
    }

}