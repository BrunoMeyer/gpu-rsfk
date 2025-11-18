#include "gpu-utils.cu"

__global__
void assign_label(float* dataset, uint dataset_size, 
        uint dim, uint k, 
        uint* labels, int* g_labels_count,
        float* centroids, float* g_centroid_change,
        float* lower_bound, float* upper_bound,
        uint t_groups, 
        uint* group_filter, float* group_lowerbounds,
        uint* group_filter_cents, uint* group_filter_location,
        uint use_shared_memory){
   
    int* label_count = g_labels_count;

    // initialize variables
    uint warpIdx = threadIdx.x / warpSize;
    uint laneIdx = threadIdx.x % warpSize;
    uint nwarps = blockDim.x / warpSize;

    // float* centroid_change = &g_centroid_change[blockIdx.x*dim*k*nwarps + warpIdx*dim*k];
    float* centroid_change = g_centroid_change;

    // for(uint i = warpIdx+blockIdx.x*nwarps; i < dataset_size; i += nwarps*blockDim.x){
    //     if(lower_bound[i] > upper_bound[i])
    //         continue;

    for(uint ii = (warpIdx+blockIdx.x*nwarps)*warpSize; ii < dataset_size; ii += nwarps*gridDim.x*warpSize){
        // GLOBAL FILTER
        uint to_be_updated = 33;
        if(lower_bound[ii+laneIdx] < upper_bound[ii+laneIdx]){
            to_be_updated=laneIdx;
        }
        uint next_update = __reduce_min_sync(0xffffffff, to_be_updated);
        while(next_update < 33){
            uint i = ii+next_update;

            uint cent=labels[i];
            ////////////////////////
            // CALCULATE DISTANCE //
            ////////////////////////
            float4 a,b;
            float s = 0.0f;
            uint nf = dim/4;
            for(uint d=laneIdx; d < nf; d+=warpSize){
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
            //////////////////////
            //////////////////////
            upper_bound[i] = new_dist;
            // float new_dist = upper_bound[i];
            if(lower_bound[i] < new_dist){ 
                //continue to group filter
                // uint min_update_count = 0;

                uint near_cent = labels[i];

                // #define SUPER_DEBUG 1
                #if SUPER_DEBUG
                if(near_cent >= k){
                    printf("Error: point %d has invalid label %d\n", i, near_cent);
                    return;
                }
                #endif

                uint near_cent_group = group_filter[near_cent];
                float min_dist = new_dist; 
                float secmin_dist = MAX_FLOAT;
            
                // variables to update group filter, if the nearest centroid group change
                float old_group_min_dist = new_dist;
                uint old_near_cent_group = near_cent_group;

                //GROUP FILTER
                for(uint g = 0; g < t_groups; g++){
                    if(group_lowerbounds[i*t_groups+g] > upper_bound[i])
                        continue;

                    float group_min_dist = MAX_FLOAT;
                    float group_sec_min_dist = MAX_FLOAT;
                        
                    for(uint j = group_filter_location[g]; j < group_filter_location[g+1]; j++){
                        uint cent = group_filter_cents[j];

                        #if SUPER_DEBUG
                        if(cent >= k || j >= k || j < 0){
                            printf("Error: group filter centroid %d at location %d is invalid in group %d\n", cent, j, g);
                            return;
                        }
                        
                        #endif

                        //LOCAL FILTER
                        // if(group_lowerbounds[i*t_groups+g] > secmin_dist)
                        //     continue;

                        ////////////////////////
                        // CALCULATE DISTANCE //
                        ////////////////////////
                        float4 a,b;
                        float s = 0.0f;
                        uint nf = dim/4;
                        for(uint d=laneIdx; d < nf; d+=warpSize){
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

                        if(new_dist < secmin_dist){
                            if(new_dist < min_dist){
                                secmin_dist = min_dist;
                                min_dist = new_dist;
                                near_cent=cent;
                                near_cent_group=g;
                            }
                            else{
                                if(near_cent!=cent){
                                    secmin_dist = new_dist;
                                }
                            }
                        }
                        if(new_dist < group_sec_min_dist){
                            if(new_dist < group_min_dist){
                                group_sec_min_dist=group_min_dist;
                                group_min_dist=new_dist;
                            }
                            else{
                                group_sec_min_dist=new_dist;
                            }
                        }
                    }
                    if(g != near_cent_group){
                        group_lowerbounds[i*t_groups+g] = group_min_dist;
                    } else {
                        //The group that has the nearest centroid must be updated with the second minimum distance
                        group_lowerbounds[i*t_groups+g] = group_sec_min_dist;
                        
                        //old group had the second min dist
                        // for this reason, it needs to be updated too
                        group_lowerbounds[i*t_groups+old_near_cent_group] = old_group_min_dist; 
                        
                        //store group min dist in case the nearest centroid group changes
                        old_group_min_dist = group_min_dist; 
                        old_near_cent_group = near_cent_group;
                    }
                }
                lower_bound[i]=secmin_dist;
                upper_bound[i]=min_dist;
                if(near_cent != labels[i]){
                    uint oldlabel=labels[i];
                    
                    if(laneIdx == 0){
                        labels[i] = near_cent;
                        atomicAdd(&label_count[near_cent],1);
                        atomicSub(&label_count[oldlabel],1);
                    }
                    // calculate the partial update for the new centroid
                    ///////////////////////////
                    // SUM POINT TO CENTROID //
                    ///////////////////////////
                    for(uint l=laneIdx; l < dim; l+=warpSize){
                        atomicAdd(&centroid_change[near_cent*dim+l], dataset[i*dim+l]);
                        // centroid_change[near_cent*dim+l] += dataset[i*dim+l];
                    }
                    for(uint l=laneIdx; l < dim; l+=warpSize){
                        atomicAdd(&centroid_change[oldlabel*dim+l], -dataset[i*dim+l]);
                        // centroid_change[oldlabel*dim+l] -= dataset[i*dim+l];
                    }
                }
            }   
            if(to_be_updated == next_update)
                to_be_updated=33;
            next_update = __reduce_min_sync(0xffffffff, to_be_updated);
        }
    }
}


// ICPADS24 VERSION
// __global__
// void assign_label(float* dataset, uint dataset_size, 
//         uint dim, uint k, 
//         uint* labels, int* global_labels_partial_count,
//         float* centroids, float* global_new_centroids,
//         float* lower_bound, float* upper_bound,
//         float* global_centroid_shift, float* global_max_centroid_shift,
//         uint t_groups, 
//         uint* group_filter, float* group_lowerbounds,
//         uint* group_filter_cents, uint* group_filter_location,
//         uint use_shared_memory){
   
//     int* label_count = &global_labels_partial_count[blockIdx.x*k];
//     float* centroid_shift;

//     extern __shared__ uint sm[];
//     if(use_shared_memory){
//         centroid_shift=(float*)sm;
//         for(int i = threadIdx.x; i < k; i+=blockDim.x){
//             centroid_shift[i] = global_centroid_shift[i];
//         }
//     }
//     else
//         centroid_shift=global_centroid_shift;

//     // initialize variables
//     uint warpIdx = threadIdx.x / WARP_SIZE;
//     uint laneIdx = threadIdx.x % WARP_SIZE;
//     uint nwarps = blockDim.x / WARP_SIZE;

//     float* centroid_change = &global_new_centroids[blockIdx.x*dim*k*nwarps + warpIdx*dim*k];
//     // float max_centroid_shift = *global_max_centroid_shift;

//     // for(uint i = warpIdx+blockIdx.x*nwarps; i < dataset_size; i += nwarps*blockDim.x){
//     //     if(lower_bound[i] > upper_bound[i])
//     //         continue;

//     for(uint ii = warpIdx*WARP_SIZE+blockIdx.x*nwarps*WARP_SIZE; ii < dataset_size; ii += nwarps*blockDim.x*WARP_SIZE){
//         // GLOBAL FILTER
//         uint to_be_updated = 33;
//         if(lower_bound[ii+laneIdx] < upper_bound[ii+laneIdx]){
//             to_be_updated=laneIdx;
//         }
//         uint next_update = __reduce_min_sync(0xffffffff, to_be_updated);
//         while(next_update < 33){
//             uint i = ii+next_update;

//             uint cent=labels[i];
//             ////////////////////////
//             // CALCULATE DISTANCE //
//             ////////////////////////
//             float4 a,b;
//             float s = 0.0f;
//             uint nf = dim/4;
//             for(uint d=laneIdx; d < nf; d+=WARP_SIZE){
//                 a = reinterpret_cast<float4*>(dataset)[i*nf+d];
//                 b = reinterpret_cast<float4*>(centroids)[cent*nf+d];
//                 float4 diff;
//                 diff.x = a.x - b.x;
//                 diff.y = a.y - b.y;
//                 diff.z = a.z - b.z;
//                 diff.w = a.w - b.w;
//                 s+=diff.x*diff.x;
//                 s+=diff.y*diff.y;
//                 s+=diff.z*diff.z;
//                 s+=diff.w*diff.w;
//             }
//             s += __shfl_xor_sync( 0xffffffff, s,  1); // assuming warpSize=32
//             s += __shfl_xor_sync( 0xffffffff, s,  2); // assuming warpSize=32
//             s += __shfl_xor_sync( 0xffffffff, s,  4); // assuming warpSize=32
//             s += __shfl_xor_sync( 0xffffffff, s,  8); // assuming warpSize=32
//             s += __shfl_xor_sync( 0xffffffff, s, 16); // assuming warpSize=32	
//             float new_dist = s;
//             //////////////////////
//             //////////////////////
//             upper_bound[i] = new_dist;
//             // float new_dist = upper_bound[i];
//             if(lower_bound[i] < new_dist){ 
//                 //continue to group filter
//                 uint min_update_count = 0;

//                 uint new_nearest = labels[i];
//                 uint near_cent_group = group_filter[new_nearest];
//                 // float min_dist = upper_bound[i]; 
//                 // float min_dist = MAX_FLOAT; 
//                 float min_dist = new_dist; 
//                 // float min_dist = MAX_FLOAT; 
//                 // float secmin_dist = new_dist; 
//                 float secmin_dist = MAX_FLOAT;
//                 // secmin_dist is also the local filter, as suggested in the Yinyang K-Means paper
            
//                 // float sec_min_dist_in_near_cent_group = new_dist;
//                 float sec_min_dist_in_near_cent_group = MAX_FLOAT;

//                 //GROUP FILTER
//                 for(uint g = 0; g < t_groups; g++){
//                     if(group_lowerbounds[i*t_groups+g] > upper_bound[i])
//                         continue;
//                     uint gmin_update_count = 0;

//                 // for(uint g = 0; g < t_groups; g+=WARP_SIZE){
//                     // uint need_group_update = 33;
//                     // if(g+laneIdx < t_groups)
//                     //     if(group_lowerbounds[i*t_groups+g+laneIdx] < upper_bound[i])
//                     //         need_group_update=laneIdx;
//                     // uint next_group = __reduce_min_sync(0xffffffff, need_group_update);
//                     // while(next_group < 33){
//                         float group_min_dist = MAX_FLOAT;
//                         float group_sec_min_dist = MAX_FLOAT;
                        
//                         for(uint j = group_filter_location[g]; j < group_filter_location[g+1]; j++){
//                             uint cent = group_filter_cents[j];
//                             // uint cent = j;

//                             //LOCAL FILTER
//                             // if(group_lowerbounds[i*t_groups+g] > secmin_dist)
//                             //     continue;

//                             ////////////////////////
//                             // CALCULATE DISTANCE //
//                             ////////////////////////
//                             float4 a,b;
//                             float s = 0.0f;
//                             uint nf = dim/4;
//                             for(uint d=laneIdx; d < nf; d+=WARP_SIZE){
//                                 a = reinterpret_cast<float4*>(dataset)[i*nf+d];
//                                 b = reinterpret_cast<float4*>(centroids)[cent*nf+d];
//                                 float4 diff;
//                                 diff.x = a.x - b.x;
//                                 diff.y = a.y - b.y;
//                                 diff.z = a.z - b.z;
//                                 diff.w = a.w - b.w;
//                                 s+=diff.x*diff.x;
//                                 s+=diff.y*diff.y;
//                                 s+=diff.z*diff.z;
//                                 s+=diff.w*diff.w;
//                             }
//                             s += __shfl_xor_sync( 0xffffffff, s,  1); // assuming warpSize=32
//                             s += __shfl_xor_sync( 0xffffffff, s,  2); // assuming warpSize=32
//                             s += __shfl_xor_sync( 0xffffffff, s,  4); // assuming warpSize=32
//                             s += __shfl_xor_sync( 0xffffffff, s,  8); // assuming warpSize=32
//                             s += __shfl_xor_sync( 0xffffffff, s, 16); // assuming warpSize=32	
//                             float new_dist = s;
//                             ////////////////////////
//                             ////////////////////////

//                             if(new_dist < group_sec_min_dist){
//                                 if(new_dist < group_min_dist){
//                                     gmin_update_count+=2;
//                                     group_sec_min_dist=group_min_dist;
//                                     group_min_dist=new_dist;
//                                 }
//                                 else{
//                                     if(new_nearest!=cent){
//                                         gmin_update_count++;
//                                         group_sec_min_dist=new_dist;
//                                     }
//                                 }
//                                 if(new_dist < secmin_dist){
//                                     if(new_dist < min_dist){
//                                         min_update_count+=2;
//                                         secmin_dist = min_dist;
//                                         min_dist = new_dist;
//                                         new_nearest=cent;
//                                         near_cent_group=g;
//                                     }
//                                     else{
//                                         if(new_nearest!=cent){
//                                             min_update_count++;
//                                             secmin_dist = new_dist;
//                                         }
//                                     }
//                                 }
//                             }
//                         }
//                         //Updating group filter
//                         if(gmin_update_count > 0){
//                             group_lowerbounds[i*t_groups+g] = group_min_dist;

//                             //The group that has the nearest centroid must be updated with the second minimum distance
//                             if(gmin_update_count > 1){
//                                 if(g == near_cent_group){
//                                     sec_min_dist_in_near_cent_group = group_sec_min_dist;
//                                 }
//                             }
//                         }
//                     //     if(next_group == need_group_update)
//                     //         need_group_update=33;
//                     //     next_group = __reduce_min_sync(0xffffffff, need_group_update);
//                     // }
//                 }
//                 uint g = group_filter[new_nearest];
//                 group_lowerbounds[i*t_groups+g] = sec_min_dist_in_near_cent_group;
//                 if(min_update_count > 0){
//                     lower_bound[i]=secmin_dist;
//                     if(min_update_count > 1){
//                         uint oldlabel=labels[i];
//                         //The group that has the nearest centroid must be updated with the second minimum distance
//                         upper_bound[i]=min_dist;
//                         if(oldlabel != new_nearest){
//                             if(laneIdx == 0){
//                                 labels[i] = new_nearest;
//                                 atomicAdd(&label_count[new_nearest],1);
//                                 atomicSub(&label_count[oldlabel],1);
//                             }
//                             // calculate the partial update for the new centroid
//                             ///////////////////////////
//                             // SUM POINT TO CENTROID //
//                             ///////////////////////////
//                             for(uint l=laneIdx; l < dim; l+=WARP_SIZE){
//                                 // atomicAdd(&centroid_change[nearest_cent*dim+l], dataset[i*dim+l]);
//                                 centroid_change[new_nearest*dim+l] += dataset[i*dim+l];
//                             }
//                             for(uint l=laneIdx; l < dim; l+=WARP_SIZE){
//                                 // atomicAdd(&centroid_change[nearest_cent*dim+l], dataset[i*dim+l]);
//                                 centroid_change[oldlabel*dim+l] -= dataset[i*dim+l];
//                             }
//                         }
//                     }
//                 }
//             }   
//             if(to_be_updated == next_update)
//                 to_be_updated=33;
//             next_update = __reduce_min_sync(0xffffffff, to_be_updated);
//         }
//     }
// }