#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <float.h>

// #include "kmeans.h"
#include "defines.h"

#include "chrono.c"

#include "kernel-functions/initialize.cu"
#include "kernel-functions/assign-label.cu"
#include "kernel-functions/align-memory.cu"
#include "kernel-functions/update.cu"
#include "kernel-functions/check-convergence.cu"

#include "kernel-functions/bounds.cu"
// #include "kernel-functions/inter_centroid_distances.cu"

#include "kernel-functions/group-filter.cu"

#include "kernel-functions/kmeanspp.cu"

void kmeansGpu(float* d_dataset, uint dataset_size, 
        uint dim, uint k, 
		uint max_it, 
		uint check_method, float tolerance, 
		uint initialization_method, //0 - random, 1 - kmeans++
		uint t_groups,
		uint verbosity,
        uint* d_labels, float* d_centroids // <-- outputs
    ) {

    int devUsed = 0;
    cudaSetDevice(devUsed);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, devUsed);

    // Tries to ensure that there are at least two blocks per multiprocessor
    int nthreads = deviceProp.maxThreadsPerMultiProcessor / 2;
    if(nthreads > deviceProp.maxThreadsPerBlock) nthreads = deviceProp.maxThreadsPerBlock;
    int nblocks = deviceProp.multiProcessorCount*(deviceProp.maxThreadsPerMultiProcessor/nthreads);

	cudaError_t err = cudaSuccess;
    chronometer_t ch_kmeans_total;
    chrono_reset(&ch_kmeans_total);
	chrono_start(&ch_kmeans_total);


    // cudaFuncSetCacheConfig(assign_label, cudaFuncCachePreferShared);
	// cudaFuncSetCacheConfig(assign_label, cudaFuncCachePreferL1);

	// dim must be a multiple of 4
	uint logic_dim = dim + dim%4; 

	// t is the number of group filters
	// uint t_groups = ceil(k/(float)10);
	if(k <= t_groups){
		t_groups = k;
	}

	// uint t_groups = ceil(k/(float)5); 
	// uint t_groups = 256; 


	uint use_shared_memory_assign_label = 0;
	uint sm_size_assign_label = 0;
	
	sm_size_assign_label = k*sizeof(float);
	use_shared_memory_assign_label = 1;
	if(sm_size_assign_label > (MAX_SM_PER_MP/(BLOCKS_PER_MP))){
		use_shared_memory_assign_label = 0;
		sm_size_assign_label = 0;
	}
	
	uint use_shared_memory_update = 1;
	uint sm_size_update = sizeof(float)*logic_dim;
	if(sm_size_update + sizeof(uint) > MAX_SM_PER_BLOCK){
		use_shared_memory_assign_label = 0;
		sm_size_update = 0;
	}

	uint n_blocks_check = ceil((float)(k*logic_dim) / (float)nthreads);



	uint n_threads_update = nthreads;
	if(logic_dim < nthreads){
		if(logic_dim < 32)
			n_threads_update = 32;
		else
			n_threads_update = logic_dim;
	}

	uint n_threads_check = nthreads;
	if(logic_dim < nthreads){
			n_threads_check = logic_dim;
	}

	//=========================
	//     ALLOCATE MEMORY
	//=========================
    chronometer_t ch_allocate;
    chrono_reset(&ch_allocate);
    chrono_start(&ch_allocate);

	int *d_labels_change_count = NULL;
	err = cudaMalloc((void **)&d_labels_change_count, sizeof(int)*k*nblocks);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to allocate device vector d_labels_change_count (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	uint *d_old_labels_count = NULL;
	err = cudaMalloc((void **)&d_old_labels_count, sizeof(uint)*k);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to allocate device vector d_old_labels_count (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	float *d_new_centroids = NULL;
	err = cudaMalloc((void **)&d_new_centroids, sizeof(float)*k*logic_dim);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to allocate device vector d_new_centroids (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	float *d_partial_centroids = NULL;
	err = cudaMalloc((void **)&d_partial_centroids, sizeof(float)*k*logic_dim*nblocks*N_WARPS);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to allocate device vector d_partial_centroids (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	float *d_sqrdNormError = NULL;
	err = cudaMalloc((void **)&d_sqrdNormError, sizeof(float));
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to allocate device vector d_sqrdNormError (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	float *d_upperbounds = NULL;
	err = cudaMalloc((void **)&d_upperbounds, sizeof(float)*dataset_size);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to allocate device vector d_upperbounds (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	float *d_lowerbounds = NULL;
	err = cudaMalloc((void **)&d_lowerbounds, sizeof(float)*dataset_size);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to allocate device vector d_lowerbounds (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	float *d_inter_cent_dist = NULL;
	err = cudaMalloc((void **)&d_inter_cent_dist, sizeof(float)*k);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to allocate device vector d_inter_cent_dist (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	float *d_centroid_shift = NULL;
	err = cudaMalloc((void **)&d_centroid_shift, sizeof(float)*k);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to allocate device vector d_centroid_shift (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	float *d_max_centroid_shift = NULL;
	err = cudaMalloc((void **)&d_max_centroid_shift, sizeof(float));
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to allocate device vector d_max_centroid_shift (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	float *d_inertia = NULL;
	err = cudaMalloc((void **)&d_inertia, sizeof(float));
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to allocate device vector d_inertia (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	uint *d_reassignments = NULL;
	err = cudaMalloc((void **)&d_reassignments, sizeof(uint));
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to allocate device vector d_reassignments (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	uint *d_group_filter_labels = NULL;
	err = cudaMalloc((void **)&d_group_filter_labels, sizeof(uint)*k);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to allocate device vector d_group_filter_labels (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	uint *d_group_filter_cents = NULL;
	err = cudaMalloc((void **)&d_group_filter_cents, sizeof(uint)*k);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to allocate device vector d_group_filter_cents (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	uint *d_group_filter_locs = NULL;
	err = cudaMalloc((void **)&d_group_filter_locs, sizeof(uint)*(t_groups+1));
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to allocate device vector d_group_filter_locs (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	float *d_group_lowerbounds = NULL;
	err = cudaMalloc((void **)&d_group_lowerbounds, sizeof(float)*t_groups*dataset_size);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to allocate device vector d_group_lowerbounds (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	float *d_max_group_shift = NULL;
	err = cudaMalloc((void **)&d_max_group_shift, sizeof(float)*t_groups);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to allocate device vector d_max_group_shift (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	uint *d_groups_with_only_one_element = NULL;
	err = cudaMalloc((void **)&d_groups_with_only_one_element, sizeof(uint));
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to allocate device vector d_groups_with_only_one_element (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

    chrono_stop(&ch_allocate);

	//=========================
	//	ALINGN DATA
	//=========================
    chronometer_t ch_align_mem;
    chrono_reset(&ch_align_mem);
    chrono_start(&ch_align_mem);
	if(dim != logic_dim){

		float *d_aligned_dataset = NULL; 
		err = cudaMalloc((void **)&d_aligned_dataset, sizeof(float)*logic_dim*dataset_size);
		if (err != cudaSuccess){
			fprintf(stderr, "Failed to allocate device vector d_aligned_dataset (error code %s)!\n", cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}

		align_memory<<<ceil((float)dataset_size/(float)nthreads),nthreads>>>(d_dataset,d_aligned_dataset, dataset_size,dim,logic_dim);

		err = cudaFree(d_dataset);
		if (err != cudaSuccess){
			fprintf(stderr, "Failed to free device vector d_dataset (error code %s)!\n", cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}
		d_dataset = d_aligned_dataset;
	}
    chrono_stop(&ch_align_mem);
	// printf("ch_align_mem time: %.3f milliseconds.\n",((chrono_gettotal(&ch_align_mem))/1000)/1000.0);

    //=================
    //  START COMPUTE
    //=================
    chronometer_t ch_init, ch_label, ch_update, ch_check;
    chronometer_t ch_init_bounds, ch_first_update;
    chrono_reset(&ch_init);
    chrono_reset(&ch_label);
    chrono_reset(&ch_update);
    chrono_reset(&ch_check);
    chrono_reset(&ch_init_bounds);
    chrono_reset(&ch_first_update);

	//---------------------
	//INITIALIZING


	chrono_start(&ch_init);
	if( initialization_method == 0){
		if(verbosity > 1){
			printf("//////////////////////\n");
			printf("//      RANDOM      //\n");
			printf("//////////////////////\n");
		} 

			initialize<<<k,nthreads>>>(
				d_dataset,dataset_size,logic_dim,
				d_centroids);
			cudaDeviceSynchronize();
		
	}
	else if( initialization_method == 1){
		/////////////////////////
		//      K-MEANS++      //
		/////////////////////////
		if(verbosity > 1){
			printf("/////////////////////////\n");
			printf("//      K-MEANS++      //\n");
			printf("/////////////////////////\n");
		}
		// ALLOC MEMORY
		uint *d_chosen_centroids = NULL; 
		err = cudaMalloc((void **)&d_chosen_centroids, sizeof(uint)*k);
		if (err != cudaSuccess){
			fprintf(stderr, "Failed to allocate device vector d_chosen_centroids (error code %s)!\n", cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}
		float *d_max_min_cent_dist = NULL; 
		err = cudaMalloc((void **)&d_max_min_cent_dist, sizeof(float)*nblocks*N_WARPS);
		if (err != cudaSuccess){
			fprintf(stderr, "Failed to allocate device vector d_max_min_cent_dist (error code %s)!\n", cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}
		uint *d_candidates_to_nextcent = NULL; 
		err = cudaMalloc((void **)&d_candidates_to_nextcent, sizeof(uint)*nblocks*N_WARPS);
		if (err != cudaSuccess){
			fprintf(stderr, "Failed to allocate device vector d_candidates_to_nextcent (error code %s)!\n", cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}
		//---
		//initialize first centroid at random and set bounds as max float
		setMaxFloat<<<ceil((float)dataset_size/(float)nthreads),nthreads>>>(d_upperbounds,dataset_size);
		setMaxFloat<<<ceil((float)k*dataset_size/(float)nthreads),nthreads>>>(d_lowerbounds,dataset_size);
		initialize_first_cent_kmeanspp<<<1,MAX_THREADS>>>(
			d_dataset,dataset_size,logic_dim,
			d_centroids, d_chosen_centroids);
		cudaDeviceSynchronize();
		gpuErrchk( cudaPeekAtLastError() );

		for(uint i = 1; i < k; i++){
			find_new_centroid_kmeanspp<<<nblocks,nthreads>>>(
				d_dataset,dataset_size,
				d_centroids,i,
				logic_dim,
				d_labels,
				d_upperbounds,d_lowerbounds,
				d_chosen_centroids,
				d_candidates_to_nextcent, d_max_min_cent_dist
			);
			cudaDeviceSynchronize();
			gpuErrchk( cudaPeekAtLastError() );

			append_centroid_kmeanspp<<<1,MAX_THREADS>>>(
				d_dataset,dataset_size,
				d_centroids,i,
				logic_dim,
				d_chosen_centroids,
				d_candidates_to_nextcent, d_max_min_cent_dist
			);
			cudaDeviceSynchronize();
			gpuErrchk( cudaPeekAtLastError() );
		}

		label_last_centroid_kmeanspp<<<nblocks,nthreads>>>(
			d_dataset,dataset_size,
			d_centroids,k,
			logic_dim,
			d_labels,
			d_upperbounds,d_lowerbounds
		);
		cudaDeviceSynchronize();
		gpuErrchk( cudaPeekAtLastError() );

		cudaMemset(d_labels_change_count, 0, sizeof(uint)*k*nblocks);	
		cudaMemset(d_new_centroids, 0, sizeof(float)*k*logic_dim);	
		sum_all_points_to_centroid<<<ceil(dataset_size*logic_dim/(float)nthreads),nthreads>>>(
			d_dataset,dataset_size,
			d_centroids,k,
			logic_dim,
			d_labels,
			d_labels_change_count,
			d_new_centroids
		);
		cudaDeviceSynchronize();
		gpuErrchk( cudaPeekAtLastError() );

		divide_sum_by_count<<<ceil(k*logic_dim/(float)nthreads),nthreads>>>(
			d_new_centroids, k, logic_dim,
			d_labels_change_count,
			d_old_labels_count
		);
		cudaDeviceSynchronize();
		gpuErrchk( cudaPeekAtLastError() );

		uint n_threads_calculate_shift = nthreads;
		if(logic_dim/4 < nthreads){
			if(logic_dim/4 < 32)
				n_threads_calculate_shift = 32;
			else
				n_threads_calculate_shift = logic_dim/4;
		}
		cudaMemset(d_max_centroid_shift, 0, sizeof(float));	
		calculate_centroid_shift<<<k,n_threads_calculate_shift>>>(
			d_centroids,
			d_new_centroids, 
			logic_dim,k,
			d_centroid_shift,d_max_centroid_shift
		);
		cudaDeviceSynchronize();
		gpuErrchk( cudaPeekAtLastError() );

		update_global_bounds<<<nblocks,nthreads>>>(
			d_labels,
			dataset_size,
			d_lowerbounds,d_upperbounds,
			d_centroid_shift,d_max_centroid_shift
		);      
		cudaDeviceSynchronize();
		gpuErrchk( cudaPeekAtLastError() );

		cudaMemset(d_group_lowerbounds,0, sizeof(float)*t_groups*dataset_size);
		float* d_old_centroids = d_centroids;
		d_centroids = d_new_centroids;
		d_new_centroids = d_old_centroids;

		err = cudaFree(d_chosen_centroids);
		if (err != cudaSuccess){
			fprintf(stderr, "Failed to free device vector d_chosen_centroids (error code %s)!\n", cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}
		err = cudaFree(d_max_min_cent_dist);
		if (err != cudaSuccess){
			fprintf(stderr, "Failed to free device vector d_max_min_cent_dist (error code %s)!\n", cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}
		err = cudaFree(d_candidates_to_nextcent);
		if (err != cudaSuccess){
			fprintf(stderr, "Failed to free device vector d_candidates_to_nextcent (error code %s)!\n", cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}
	}
	// else{
	// 	cudaMemcpy(d_new_centroids, centroids,  sizeof(float)*dim*k, cudaMemcpyHostToDevice);	
	// 	if(dim != logic_dim){
	// 		align_memory<<<ceil((float)k/(float)nthreads),nthreads>>>(d_new_centroids,d_centroids, k,dim,logic_dim);
	// 		cudaDeviceSynchronize();
	// 		gpuErrchk( cudaPeekAtLastError() );
	// 	}
	// }
	chrono_stop(&ch_init);

	////////////////////////////
	//      GROUP FILTER      //
	////////////////////////////
	//init group filter


	// ALLOC MEMORY
	float *d_group_centroids = NULL; 
	err = cudaMalloc((void **)&d_group_centroids, sizeof(float)*t_groups*logic_dim);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to allocate device vector d_group_centroids (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	cudaMemcpy(d_group_centroids, d_centroids, sizeof(float)*t_groups*logic_dim,cudaMemcpyDeviceToDevice);

	uint removed_groups;

    chronometer_t ch_creating_group_filter;
    chrono_reset(&ch_creating_group_filter);
    chrono_start(&ch_creating_group_filter);
	do {
		// cudaMemcpy(d_group_centroids, d_centroids, sizeof(float)*t_groups*dim,cudaMemcpyDeviceToDevice);
		if(verbosity > 2)
			printf("Group filter size = %d\n",t_groups);
		for(uint i = 0; i < 5; i++){
			group_filter_assignment<<<nblocks,nthreads>>>(d_centroids, k, 
				logic_dim, t_groups, 
				d_group_centroids, d_group_filter_labels);
			cudaDeviceSynchronize();
			gpuErrchk( cudaPeekAtLastError() );

			group_filter_update<<<t_groups,nthreads,logic_dim*sizeof(float)>>>(d_centroids, k, 
					logic_dim,
					d_group_centroids, d_group_filter_labels);
			cudaDeviceSynchronize();
			gpuErrchk( cudaPeekAtLastError() );


		}
		organize_group_filter<<<1,MAX_THREADS,(t_groups*3+1)*sizeof(uint)>>>( 
			d_group_centroids, k, logic_dim, 
			t_groups,
			d_group_filter_labels, 
			d_group_filter_cents, d_group_filter_locs,
			d_groups_with_only_one_element
			// ,(uint*)d_new_centroids
		);
		cudaDeviceSynchronize();
		gpuErrchk( cudaPeekAtLastError() );

		removed_groups = 0;
		cudaMemcpy(&removed_groups, d_groups_with_only_one_element, sizeof(uint),cudaMemcpyDeviceToHost);
		// if(removed_groups)
		// 	printf("ERROR in %s %d: invalid groups = %d)\n",__FILE__, __LINE__,removed_groups);
		t_groups-=removed_groups;
	} while(removed_groups);
    chrono_stop(&ch_creating_group_filter);

	err = cudaFree(d_group_centroids);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to free device vector d_dataset (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	// printf("Group filter size = %d\n",t_groups);



	#ifdef ORGANIZE_CENTS_IN_MEMORY
		chronometer_t ch_organize_cents;
		chrono_reset(&ch_organize_cents);
		chrono_start(&ch_organize_cents);

		float *d_organized_centroids = NULL; 
		err = cudaMalloc((void **)&d_organized_centroids, sizeof(float)*logic_dim*dataset_size);
		if (err != cudaSuccess){
			fprintf(stderr, "Failed to allocate device vector d_organized_centroids (error code %s)!\n", cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}

		organize_cents_in_memory<<<nblocks,nthreads>>>(d_centroids, d_organized_centroids, k, logic_dim,
			d_group_filter_labels, 
			d_group_filter_cents, d_group_filter_locs);

		err = cudaFree(d_centroids);
		if (err != cudaSuccess){
			fprintf(stderr, "Failed to free device vector d_dataset (error code %s)!\n", cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}
		d_centroids = d_organized_centroids;

		chrono_stop(&ch_organize_cents);
		if(verbosity > 2)
			printf("ch_organize_cents time: %.3f milliseconds.\n",((chrono_gettotal(&ch_organize_cents))/1000)/1000.0);
	#endif
	if( initialization_method == 0){


		chrono_start(&ch_init_bounds);
		
			cudaMemset(d_labels_change_count, 0, sizeof(uint)*k*nblocks);	
			cudaMemset(d_partial_centroids, 0, sizeof(float)*k*logic_dim*nblocks*N_WARPS);	
			init_bounds_and_assign_labels<<<nblocks,nthreads>>>(
				d_dataset,dataset_size,
				d_centroids,k,
				logic_dim,
				d_labels, d_labels_change_count,
				d_upperbounds,d_lowerbounds,
				t_groups,
				d_group_filter_labels, d_group_lowerbounds,
				d_partial_centroids
				,d_group_filter_cents, d_group_filter_locs);
			// );
			cudaDeviceSynchronize();
		
		chrono_stop(&ch_init_bounds);
		gpuErrchk( cudaPeekAtLastError() );


		chrono_start(&ch_first_update);
		
			cudaMemset(d_max_centroid_shift, 0, sizeof(float));	
			cudaMemset(d_reassignments, 0, sizeof(uint));
			cudaMemset(d_max_group_shift, 0, sizeof(float)*t_groups);	
			update<<<k,n_threads_update,sm_size_update>>>(
				d_centroids,
				logic_dim,k,
				d_labels, d_labels_change_count,
				d_old_labels_count,
				d_new_centroids, d_partial_centroids,
				d_centroid_shift,d_max_centroid_shift, 
				d_group_filter_labels,
				d_max_group_shift,
				d_reassignments,
				use_shared_memory_update);
			cudaDeviceSynchronize();

			update_bounds<<<nblocks,nthreads>>>(
				dataset_size,
				logic_dim,k,
				d_labels,
				d_lowerbounds,d_upperbounds,
				d_centroid_shift,d_max_centroid_shift, 
				t_groups,
				d_group_lowerbounds,
				d_max_group_shift);      
			cudaDeviceSynchronize();
		chrono_stop(&ch_first_update);
		gpuErrchk( cudaPeekAtLastError() );

		float* d_old_centroids = d_centroids;
		d_centroids = d_new_centroids;
		d_new_centroids = d_old_centroids;

	}

    uint n_it = 0;
	float sqrdNormError = MAX_FLOAT;
    while(max_it > n_it){
		n_it++;

        long long time_start_label = chrono_gettotal(&ch_label);
        chrono_start(&ch_label);
		cudaMemset(d_partial_centroids, 0, sizeof(float)*k*logic_dim*nblocks*N_WARPS);	

            assign_label<<<nblocks,nthreads,sm_size_assign_label>>>(
                d_dataset,dataset_size,
				logic_dim,k,
                d_labels, d_labels_change_count,
                d_centroids, d_partial_centroids,
				d_lowerbounds,d_upperbounds,
				d_centroid_shift,d_max_centroid_shift,
				t_groups,
				d_group_filter_labels, d_group_lowerbounds,
				d_group_filter_cents, d_group_filter_locs,
				use_shared_memory_assign_label);
            cudaDeviceSynchronize();
        

        chrono_stop(&ch_label);
		gpuErrchk( cudaPeekAtLastError() );


        long long update_start_time = chrono_gettotal(&ch_update);
        chrono_start(&ch_update);

			cudaMemset(d_max_centroid_shift, 0, sizeof(float));	
			cudaMemset(d_reassignments, 0, sizeof(uint));
			cudaMemset(d_max_group_shift, 0, sizeof(float)*t_groups);	
            update<<<k,n_threads_update,sm_size_update>>>(
                d_centroids,
				logic_dim,k,
                d_labels, d_labels_change_count,
				d_old_labels_count,
                d_new_centroids, d_partial_centroids,
				d_centroid_shift,d_max_centroid_shift, 
				d_group_filter_labels,
				d_max_group_shift,
				d_reassignments,
				use_shared_memory_update);
            cudaDeviceSynchronize();

			update_bounds<<<nblocks,nthreads>>>(
				dataset_size,
				logic_dim,k,
				d_labels,
				d_lowerbounds,d_upperbounds,
				d_centroid_shift,d_max_centroid_shift, 
				t_groups,
				d_group_lowerbounds,
				d_max_group_shift);      
            cudaDeviceSynchronize();
        chrono_stop(&ch_update);
		gpuErrchk( cudaPeekAtLastError() );

		chrono_start(&ch_check);

			cudaMemset(d_sqrdNormError, 0, sizeof(float));	
            check_convergence<<<k,n_threads_check>>>(
                d_centroids,d_new_centroids,logic_dim,k,d_sqrdNormError);
            cudaDeviceSynchronize();	
        
			cudaMemset(d_inertia, 0, sizeof(float));	
			sumReduce<<<nblocks,nthreads>>>(d_upperbounds,dataset_size,d_inertia);
			cudaDeviceSynchronize();

        chrono_stop(&ch_check);
		gpuErrchk( cudaPeekAtLastError() );

		float* d_old_centroids = d_centroids;
		d_centroids = d_new_centroids;
		d_new_centroids = d_old_centroids;

		float inertia;
		cudaMemcpy(&inertia, d_inertia, sizeof(float), cudaMemcpyDeviceToHost);

		uint reassignments;
		cudaMemcpy(&reassignments, d_reassignments, sizeof(uint), cudaMemcpyDeviceToHost);
		cudaMemcpy(&sqrdNormError, d_sqrdNormError, sizeof(float), cudaMemcpyDeviceToHost);
		if(verbosity > 2){
			long long time_final_label = chrono_gettotal(&ch_label);
	        printf("labeling time: %.3f milliseconds.\n",(time_final_label-time_start_label)/1000/1000.0); // total time spent running knn kernel
			long long update_final_time = chrono_gettotal(&ch_update);
	        printf("updating time: %.3f milliseconds.\n",(update_final_time-update_start_time)/1000/1000.0); // total time spent running knn kernel
			printf("upper bound sum=%.3f \n",inertia);
			printf("sqrdNormError=%f \n",sqrdNormError);
			printf("reassignments=%.3f%% (%u)\n",reassignments/(float)dataset_size*100,reassignments);
		}

		if(check_method == 1 && sqrdNormError <= tolerance)
			break;
    	if(check_method == 2 && reassignments/(float)dataset_size <= tolerance)
			break;
    }


	unalign_memory<<<ceil((float)k/(float)nthreads),nthreads>>>(d_centroids,d_new_centroids, k,dim,logic_dim);
	cudaDeviceSynchronize();

    //==================
	//  DEALLOC MEMORY
	//==================
    chronometer_t ch_dealloc_mem;
    chrono_reset(&ch_dealloc_mem);
    chrono_start(&ch_dealloc_mem);

	err = cudaFree(d_new_centroids);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to free device vector d_new_centroids (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaFree(d_labels_change_count);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to free device vector d_labels_change_count (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	err = cudaFree(d_old_labels_count);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to free device vector d_old_labels_count (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaFree(d_partial_centroids);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to free device vector d_partial_centroids (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaFree(d_sqrdNormError);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to free device vector d_sqrdNormError (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaFree(d_upperbounds);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to free device vector d_upperbounds (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaFree(d_lowerbounds);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to free device vector d_lowerbounds (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaFree(d_inter_cent_dist);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to free device vector d_inter_cent_dist (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	
	err = cudaFree(d_centroid_shift);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to free device vector d_centriod_shift (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaFree(d_max_centroid_shift);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to free device vector d_max_centriod_shift (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaFree(d_inertia);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to free device vector d_inertia (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaFree(d_reassignments);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to free device vector d_reassignments (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	
	err = cudaFree(d_group_filter_labels);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to free device vector d_group_filter_labels (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaFree(d_group_filter_cents);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to free device vector d_group_filter_cents (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	
	err = cudaFree(d_group_filter_locs);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to free device vector d_group_filter_locs (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaFree(d_group_lowerbounds);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to free device vector d_group_lowerbounds (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	
	err = cudaFree(d_max_group_shift);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to free device vector d_max_group_shift (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaFree(d_groups_with_only_one_element);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to free device vector d_groups_with_only_one_element (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}


    chrono_stop(&ch_dealloc_mem);

	chrono_stop(&ch_kmeans_total);
	//===============
	//  TIME REPORT
	//===============
    long long init_time = chrono_gettotal(&ch_init);
    long long init_bound_time = chrono_gettotal(&ch_init_bounds);
    long long label_time = chrono_gettotal(&ch_label);
    long long update_time = chrono_gettotal(&ch_update);
    long long first_update_time = chrono_gettotal(&ch_first_update);
    long long alloc_time = chrono_gettotal(&ch_allocate);
    long long dealloc_time = chrono_gettotal(&ch_dealloc_mem);
    long long total_time = chrono_gettotal(&ch_kmeans_total);
    long long check_time = chrono_gettotal(&ch_check);
	long long kmeans_time = total_time-alloc_time-dealloc_time;

	if(verbosity > 2){
		printf("==================\n");
		printf("Total time: %.3f milliseconds\n",total_time/1000/1000.0);
		printf("total of %u iterations.\n",n_it);
		printf("time spent on:\n");
		printf("  - init centroids: %.3f milliseconds\n",init_time/1000/1000.0); // total time spent running knn kernel
		printf("  - creating group filter: %.3f milliseconds.\n",((chrono_gettotal(&ch_creating_group_filter))/1000)/1000.0);
		printf("  - init bounds: %.3f milliseconds\n",init_bound_time/1000/1000.0); // total time spent running knn kernel
		printf("  - first update: %.3f milliseconds\n",first_update_time/1000/1000.0); // total time spent running knn kernel
		printf("  - labeling: %.3f milliseconds\n",label_time/1000/1000.0);
		if(n_it > 0) printf("      - average time per iteration: %.3f milliseconds\n",label_time/(1000*n_it)/1000.0);
		printf("  - updating: %.3f milliseconds\n",update_time/1000/1000.0);
		if(n_it > 0) printf("      - average time per iteration: %.3f milliseconds\n",update_time/(1000*n_it)/1000.0);
		printf("  - checking convergence: %.3f milliseconds\n",check_time/1000/1000.0);
		if(n_it > 0) printf("      - average time per iteration: %.3f milliseconds\n",check_time/(1000*n_it)/1000.0);
		printf("  - allocating memory: %.3f milliseconds\n",(alloc_time)/1000/1000.0);
		printf("  - deallocating memory: %.3f milliseconds\n",(dealloc_time)/1000/1000.0);
	}
	if(verbosity > 1){
		printf("==================\n");
		printf("total time: %.3f milliseconds\n",kmeans_time/1000/1000.0);
		printf("%u iterations.\n",n_it);
		printf("init bounds: %.3f milliseconds\n",(init_time+init_bound_time+chrono_gettotal(&ch_creating_group_filter))/1000/1000.0); // total time spent running knn kernel
		printf("iteration time: %.3f milliseconds\n",(check_time+label_time+update_time)/1000/1000.0);
		if(n_it > 0){
			printf("time per it: %.3f milliseconds\n",(check_time+label_time+update_time)/(1000*(n_it))/1000.0);
			printf("total time / n it: %.3f milliseconds\n",(kmeans_time)/(1000*(n_it))/1000.0);

		}
		printf("===================\n");
	}
	if(verbosity == 1){
		printf("%.3f -n%u ",((kmeans_time)/1000)/(float)1000,n_it);
	}

}