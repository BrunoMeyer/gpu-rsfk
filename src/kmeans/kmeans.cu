#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <float.h>

// #include "kmeans.h"
#include "defines.h"

#include "chrono.c"
#include "cpu-utils.c"

#include "kernel-functions/initialize.cu"
#include "kernel-functions/assign-label.cu"
#include "kernel-functions/align-memory.cu"
#include "kernel-functions/update.cu"
#include "kernel-functions/check-convergence.cu"

#include "kernel-functions/bounds.cu"
// #include "kernel-functions/inter_centroid_distances.cu"

#include "kernel-functions/group-filter.cu"

#include "kernel-functions/kmeanspp.cu"


void write_data_from_device(float* d_data, uint n, uint d, char* filename) {

	thrust::host_vector<float> h_data(d*n);
	cudaMemcpy(thrust::raw_pointer_cast(h_data.data()), d_data, sizeof(float)*d*n, cudaMemcpyDeviceToHost);
	write_data(
		filename,
		n,
		d,
		thrust::raw_pointer_cast(h_data.data()));
}

void write_data_from_device(int* d_data, uint n, uint d, char* filename) {

	thrust::host_vector<int> h_data(d*n);
	cudaMemcpy(thrust::raw_pointer_cast(h_data.data()), d_data, sizeof(int)*d*n, cudaMemcpyDeviceToHost);
	write_data(
		filename,
		n,
		d,
		thrust::raw_pointer_cast(h_data.data()));
}

/////////////////////////
//      K-MEANS++      //
/////////////////////////
void kmeanspp(float* d_dataset, uint dataset_size, 
        uint dim, uint k, 
		uint verbosity,
		//OUTPUTS
		float* d_centroids,
        uint* d_labels
){

	// ALLOC MEMORY
	cudaError_t err = cudaSuccess;
	int devUsed = 0;
	cudaSetDevice(devUsed);
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, devUsed);

	int max_threads = deviceProp.maxThreadsPerBlock;
	if(max_threads > MAX_THREADS && verbosity){
		printf("WA in %s %d: The macro MAX_THREADS (%d) is lower than the device max threads per block (%d). \n", __FILE__,__LINE__, MAX_THREADS,max_threads);
		max_threads = MAX_THREADS;
	}
	int nthreads = deviceProp.maxThreadsPerMultiProcessor / 2;
	if(nthreads > deviceProp.maxThreadsPerBlock) nthreads = deviceProp.maxThreadsPerBlock;
	int nblocks = deviceProp.multiProcessorCount*(deviceProp.maxThreadsPerMultiProcessor/nthreads);
	int nwarps = nthreads / WARP_SIZE;

	float *d_lowerbounds = NULL;
	err = cudaMalloc((void **)&d_lowerbounds, sizeof(float)*dataset_size);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to allocate device vector d_lowerbounds (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	float *d_upperbounds = NULL;
	err = cudaMalloc((void **)&d_upperbounds, sizeof(float)*dataset_size);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to allocate device vector d_upperbounds (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	uint *d_chosen_centroids = NULL;
	err = cudaMalloc((void **)&d_chosen_centroids, sizeof(uint)*k);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to allocate device vector d_chosen_centroids (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	float *d_max_min_cent_dist = NULL; 
	err = cudaMalloc((void **)&d_max_min_cent_dist, sizeof(float)*nblocks*nwarps);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to allocate device vector d_max_min_cent_dist (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	uint *d_candidates_to_nextcent = NULL; 
	err = cudaMalloc((void **)&d_candidates_to_nextcent, sizeof(uint)*nblocks*nwarps);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to allocate device vector d_candidates_to_nextcent (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	//---
	//initialize first centroid at random and set bounds as max float
	setMaxFloat<<<ceil((float)dataset_size/(float)nthreads),nthreads>>>(d_upperbounds,dataset_size);
	setMaxFloat<<<ceil((float)dataset_size/(float)nthreads),nthreads>>>(d_lowerbounds,dataset_size);
	
	// RANDOM FIRST CENTROID 
	static unsigned long long seed = 0;
	initialize_first_cent_kmeanspp<<<1,max_threads>>>(
		d_dataset,dataset_size,dim,
		d_centroids, d_chosen_centroids, seed);
	seed+=1;

	// GET FIRST POINT TO BE THE FIRST CENTROID
	// initialize_first_cent_kmeanspp<<<1,max_threads>>>(
	// 	d_dataset,dataset_size,logic_dim,
	// 	d_centroids, d_chosen_centroids);
	
	cudaDeviceSynchronize();
	gpuErrchk( cudaPeekAtLastError() );

	for(uint n_chosen_centroids = 1; n_chosen_centroids < k; n_chosen_centroids++){
		find_new_centroid_kmeanspp<<<nblocks,nthreads>>>(
			d_dataset,dataset_size,
			d_centroids,n_chosen_centroids,
			dim,
			d_labels,
			d_upperbounds,d_lowerbounds,
			d_chosen_centroids,
			d_candidates_to_nextcent, d_max_min_cent_dist
		);
		cudaDeviceSynchronize();
		gpuErrchk( cudaPeekAtLastError() );

		//WARNING: max_threads has to be power of 2
		append_centroid_kmeanspp<<<1,max_threads>>>(
			d_dataset,dataset_size,
			d_centroids,n_chosen_centroids,
			dim,
			d_chosen_centroids,
			d_candidates_to_nextcent, d_max_min_cent_dist, nblocks*nwarps
		);
		cudaDeviceSynchronize();
		gpuErrchk( cudaPeekAtLastError() );

		#if DEBUG_KMEANSPP_WRITE_FILES 
			char filename[100];
			sprintf(filename,"./out/centroids-gr-kmeanspp-it-%03u.txt",n_chosen_centroids);
			write_data_from_device(
				d_centroids,
				n_chosen_centroids+1,
				dim,
				filename
			);

			sprintf(filename,"./out/labels-gr-kmeanspp-it-%03u.txt",n_chosen_centroids);
			write_data_from_device(
				(int*)d_labels,
				dataset_size,
				1,
				filename
			);


			sprintf(filename,"./out/upperbound-gr-kmeanspp-it-%03u.txt",n_chosen_centroids);
			write_data_from_device(
				d_upperbounds,
				dataset_size,
				1,
				filename
			);


			sprintf(filename,"./out/lowerbound-gr-kmeanspp-it-%03u.txt",n_chosen_centroids);
			write_data_from_device(
				d_lowerbounds,
				dataset_size,
				1,
				filename
			);
		#endif


	}

	#if DEBUG_KMEANSPP_WRITE_FILES 
		char filename[100];
		sprintf(filename,"./out/gr-chosen_centroids.txt",d_chosen_centroids);
		write_data_from_device(
			(int*)d_chosen_centroids,
			k,
			1,
			filename
		);
	#endif
	label_last_centroid_kmeanspp<<<nblocks,nthreads>>>(
		d_dataset,dataset_size,
		d_centroids,k,
		dim,
		d_labels,
		d_upperbounds,d_lowerbounds
	);

	cudaDeviceSynchronize();
	gpuErrchk( cudaPeekAtLastError() );

	err = cudaFree(d_lowerbounds);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to free device vector d_lowerbounds (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	err = cudaFree(d_upperbounds);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to free device vector d_upperbounds (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

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

void kmeansppAndBoundsInitialization(float* d_dataset, uint dataset_size, 
        uint dim, uint k, 
		uint max_it, 
		uint verbosity,
		//OUTPUTS
		float* d_centroids,
        uint* d_labels,
		float* d_upperbounds,
		float* d_lowerbounds,
		float* d_new_centroids,
		uint* d_labels_count,
		float* d_centroid_shift,
		float* d_max_centroid_shift
){

	if(verbosity > 2){
		printf("/////////////////////////\n");
		printf("//      K-MEANS++      //\n");
		printf("/////////////////////////\n");
	}
	// ALLOC MEMORY
	cudaError_t err = cudaSuccess;
	int devUsed = 0;
	cudaSetDevice(devUsed);
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, devUsed);

	int max_threads = deviceProp.maxThreadsPerBlock;
	if(max_threads > MAX_THREADS){
		printf("WA in %s %d: The macro MAX_THREADS (%d) is lower than the device max threads per block (%d). \n", __FILE__,__LINE__, MAX_THREADS,max_threads);
		max_threads = MAX_THREADS;
	}
	int nthreads = deviceProp.maxThreadsPerMultiProcessor / 2;
	if(nthreads > deviceProp.maxThreadsPerBlock) nthreads = deviceProp.maxThreadsPerBlock;
	int nblocks = deviceProp.multiProcessorCount*(deviceProp.maxThreadsPerMultiProcessor/nthreads);
	int nwarps = nthreads / WARP_SIZE;


	uint *d_chosen_centroids = NULL;
	err = cudaMalloc((void **)&d_chosen_centroids, sizeof(uint)*k);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to allocate device vector d_chosen_centroids (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	float *d_max_min_cent_dist = NULL; 
	err = cudaMalloc((void **)&d_max_min_cent_dist, sizeof(float)*nblocks*nwarps);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to allocate device vector d_max_min_cent_dist (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	uint *d_candidates_to_nextcent = NULL; 
	err = cudaMalloc((void **)&d_candidates_to_nextcent, sizeof(uint)*nblocks*nwarps);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to allocate device vector d_candidates_to_nextcent (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	//---
	//initialize first centroid at random and set bounds as max float
	setMaxFloat<<<ceil((float)dataset_size/(float)nthreads),nthreads>>>(d_upperbounds,dataset_size);
	setMaxFloat<<<ceil((float)dataset_size/(float)nthreads),nthreads>>>(d_lowerbounds,dataset_size);
	
	// RANDOM FIRST CENTROID 
	static unsigned long long seed = 0;
	initialize_first_cent_kmeanspp<<<1,max_threads>>>(
		d_dataset,dataset_size,dim,
		d_centroids, d_chosen_centroids, seed);
	seed+=1;

	// GET FIRST POINT TO BE THE FIRST CENTROID
	// initialize_first_cent_kmeanspp<<<1,max_threads>>>(
	// 	d_dataset,dataset_size,logic_dim,
	// 	d_centroids, d_chosen_centroids);
	
	cudaDeviceSynchronize();
	gpuErrchk( cudaPeekAtLastError() );

	for(uint n_chosen_centroids = 1; n_chosen_centroids < k; n_chosen_centroids++){
		find_new_centroid_kmeanspp<<<nblocks,nthreads>>>(
			d_dataset,dataset_size,
			d_centroids,n_chosen_centroids,
			dim,
			d_labels,
			d_upperbounds,d_lowerbounds,
			d_chosen_centroids,
			d_candidates_to_nextcent, d_max_min_cent_dist
		);
		cudaDeviceSynchronize();
		gpuErrchk( cudaPeekAtLastError() );

		//WARNING: max_threads has to be power of 2
		append_centroid_kmeanspp<<<1,max_threads>>>(
			d_dataset,dataset_size,
			d_centroids,n_chosen_centroids,
			dim,
			d_chosen_centroids,
			d_candidates_to_nextcent, d_max_min_cent_dist, nblocks*nwarps
		);
		cudaDeviceSynchronize();
		gpuErrchk( cudaPeekAtLastError() );

		#if DEBUG_KMEANSPP_WRITE_FILES 
			char filename[100];
			sprintf(filename,"./out/centroids-kmeanspp-it-%03u.txt",n_chosen_centroids);
			write_data_from_device(
				d_centroids,
				n_chosen_centroids+1,
				dim,
				filename
			);

			sprintf(filename,"./out/labels-kmeanspp-it-%03u.txt",n_chosen_centroids);
			write_data_from_device(
				(int*)d_labels,
				dataset_size,
				1,
				filename
			);


			sprintf(filename,"./out/upperbound-kmeanspp-it-%03u.txt",n_chosen_centroids);
			write_data_from_device(
				d_upperbounds,
				dataset_size,
				1,
				filename
			);


			sprintf(filename,"./out/lowerbound-kmeanspp-it-%03u.txt",n_chosen_centroids);
			write_data_from_device(
				d_lowerbounds,
				dataset_size,
				1,
				filename
			);
		#endif


	}

	#if DEBUG_KMEANSPP_WRITE_FILES 
		char filename[100];
		sprintf(filename,"./out/chosen_centroids.txt",d_chosen_centroids);
		write_data_from_device(
			(int*)d_chosen_centroids,
			k,
			1,
			filename
		);
	#endif
	
	label_last_centroid_kmeanspp<<<nblocks,nthreads>>>(
		d_dataset,dataset_size,
		d_centroids,k,
		dim,
		d_labels,
		d_upperbounds,d_lowerbounds
	);

	cudaDeviceSynchronize();
	gpuErrchk( cudaPeekAtLastError() );

	cudaMemset(d_labels_count, 0, sizeof(uint)*k);	
	cudaMemset(d_new_centroids, 0, sizeof(float)*k*dim);	
	sum_all_points_to_centroid<<<ceil(dataset_size*dim/(float)nthreads),nthreads>>>(
		d_dataset,dataset_size,
		d_centroids,k,
		dim,
		d_labels,
		d_labels_count,
		d_new_centroids
	);
	cudaDeviceSynchronize();
	gpuErrchk( cudaPeekAtLastError() );

	divide_sum_by_count<<<ceil(k*dim/(float)nthreads),nthreads>>>(
		d_new_centroids, k, dim,
		d_labels_count
	);
	cudaDeviceSynchronize();
	gpuErrchk( cudaPeekAtLastError() );

	cudaMemset(d_max_centroid_shift, 0, sizeof(float));	
	calculate_centroid_shift<<<k,nthreads>>>(
		d_centroids,
		d_new_centroids, 
		dim,k,
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

//////////////////////////////////
//////////////////////////////////
//////////////////////////////////
//////                      //////
//////      K-MEANS GPU     //////
//////                      //////
//////////////////////////////////
//////////////////////////////////
//////////////////////////////////

void kmeansGpu(float* d_dataset, uint dataset_size, 
        uint dim, uint k, 
		uint max_it, 
		uint check_method, float tolerance, 
		uint initialization_method, //0 - random, 1 - kmeans++
		uint t_groups,
		uint verbosity,
		// outputs
        uint* d_labels, 
		// optional outputs
		float* d_output_centroids = NULL,
		float* d_upperbounds = NULL,
		float* d_aligned_dataset = NULL

    ) {

    int devUsed = 0;
    cudaSetDevice(devUsed);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, devUsed);

    // Tries to ensure that there are at least two blocks per multiprocessor
	int max_threads = deviceProp.maxThreadsPerBlock;
	if(max_threads > MAX_THREADS){
		printf("WARNING: The macro MAX_THREADS (%d) is lower than the device max threads per block (%d). \n",MAX_THREADS,max_threads);
		max_threads = MAX_THREADS;
	}
    int nthreads = deviceProp.maxThreadsPerMultiProcessor / 2;
    if(nthreads > deviceProp.maxThreadsPerBlock) nthreads = deviceProp.maxThreadsPerBlock;
    int nblocks = deviceProp.multiProcessorCount*(deviceProp.maxThreadsPerMultiProcessor/nthreads);
	int nwarps = nthreads / WARP_SIZE;

	if (verbosity > 1 ){
		printf("K-Means GPU Implementation\n");
		printf("Using %d blocks of %d threads (Device: %s)\n",nblocks,nthreads,deviceProp.name);
		printf("Max threads per block: %d\n",max_threads);
		printf("Dataset size: %d, Dim: %d, K: %d\n",dataset_size,dim,k);
	}


	cudaError_t err = cudaSuccess;
    chronometer_t ch_kmeans_total;
    chrono_reset(&ch_kmeans_total);
	chrono_start(&ch_kmeans_total);


    // cudaFuncSetCacheConfig(assign_label, cudaFuncCachePreferShared);
	// cudaFuncSetCacheConfig(assign_label, cudaFuncCachePreferL1);

	// dim must be a multiple of 4
	uint logic_dim = ((dim+3)/4)*4; 

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



	float *d_centroids = d_output_centroids;
	if(d_centroids == NULL){
		err = cudaMalloc((void **)&d_centroids, sizeof(float)*k*logic_dim);
		if (err != cudaSuccess){
			fprintf(stderr, "Failed to allocate device vector d_centroids (error code %s)!\n", cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}
	}

	uint *d_labels_count = NULL;
	err = cudaMalloc((void **)&d_labels_count, sizeof(uint)*k);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to allocate device vector d_labels_count (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	int *d_labels_change_count = NULL;
	err = cudaMalloc((void **)&d_labels_change_count, sizeof(int)*k);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to allocate device vector d_labels_change_count (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	float *d_new_centroids = NULL;
	err = cudaMalloc((void **)&d_new_centroids, sizeof(float)*k*logic_dim);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to allocate device vector d_new_centroids (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}


	float *d_centroids_change = NULL;
	err = cudaMalloc((void **)&d_centroids_change, sizeof(float)*k*logic_dim);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to allocate device vector d_centroids_change (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	float *d_sqrdNormError = NULL;
	err = cudaMalloc((void **)&d_sqrdNormError, sizeof(float));
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to allocate device vector d_sqrdNormError (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// float *d_upperbounds = NULL;
	int allocated_upperbounds = 0;
	if(d_upperbounds == NULL){
		allocated_upperbounds = 1;
		err = cudaMalloc((void **)&d_upperbounds, sizeof(float)*dataset_size);
		if (err != cudaSuccess){
			fprintf(stderr, "Failed to allocate device vector d_upperbounds (error code %s)!\n", cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}
	}
	float *d_lowerbounds = NULL;
	int allocated_lowerbounds = 0;
	if(d_lowerbounds == NULL){
		allocated_lowerbounds = 1;
		err = cudaMalloc((void **)&d_lowerbounds, sizeof(float)*dataset_size);
		if (err != cudaSuccess){
			fprintf(stderr, "Failed to allocate device vector d_lowerbounds (error code %s)!\n", cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}
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
	uint *d_group_filter_cent_ids = NULL;
	err = cudaMalloc((void **)&d_group_filter_cent_ids, sizeof(uint)*k);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to allocate device vector d_group_filter_cent_ids (error code %s)!\n", cudaGetErrorString(err));
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

	// float *d_aligned_dataset = NULL; 	
	float *d_original_dataset = d_dataset;
	// #define DEBUG_PYTHON 1
	// #if DEBUG_PYTHON
		// fprintf(stderr, "KMeansGpu: Aligning dataset from dim %d to logic_dim %d\n",dim,logic_dim);
	// #endif
	int allocated_aligned_dataset = 0;
	if(dim != logic_dim){

		if(d_aligned_dataset == NULL){
			allocated_aligned_dataset = 1;
			err = cudaMalloc((void **)&d_aligned_dataset, sizeof(float)*logic_dim*dataset_size);
			if (err != cudaSuccess){
				fprintf(stderr, "Failed to allocate device vector d_aligned_dataset (error code %s)!\n", cudaGetErrorString(err));
				exit(EXIT_FAILURE);
			}
		}

		align_memory<<<ceil((float)dataset_size/(float)nthreads),nthreads>>>(d_dataset,d_aligned_dataset, dataset_size,dim,logic_dim);

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
	float* d_old_centroids = d_centroids;
	if( initialization_method == 0){
		if(verbosity > 1){
			printf("//////////////////////\n");
			printf("//      RANDOM      //\n");
			printf("//////////////////////\n");
		} 
		printf("ERROR: Random Initialization of centroids is outdated. Use KMeans++ initialization.\n");
		exit(EXIT_FAILURE);

		// initialize<<<k,nthreads>>>(
		// 	d_dataset,dataset_size,logic_dim,
		// 	d_centroids);
		// cudaDeviceSynchronize();
		
	}
	else if( initialization_method == 1){
		kmeansppAndBoundsInitialization(
			d_dataset, dataset_size,
			logic_dim, k,
			max_it,
			verbosity,
			//OUTPUTS
			d_centroids,
			d_labels,
			d_upperbounds,
			d_lowerbounds,
			d_new_centroids,
			d_labels_count,
			d_centroid_shift,
			d_max_centroid_shift
		);
		d_centroids = d_new_centroids;
		d_new_centroids = d_old_centroids;
		cudaDeviceSynchronize();
		gpuErrchk( cudaPeekAtLastError() );

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




	#if DEBUG_KMEANS_WRITE_FILES 
		write_data_from_device(
			d_centroids,
			k,
			logic_dim,
			"./out/centroids-it-000.txt"
		);

		write_data_from_device(
			(int*)d_labels,
			dataset_size,
			1,
			"./out/labels-it-000.txt"
		);

		write_data_from_device(
			d_upperbounds,
			dataset_size,
			1,
			"./out/upperbounds-it-000.txt"
		);
	#endif









	////////////////////////////
	//      GROUP FILTER      //
	////////////////////////////
	//init group filter
	cudaMemset(d_group_lowerbounds,0, sizeof(float)*t_groups*dataset_size);

	// ALLOC MEMORY
	float *d_group_centroids = NULL; 
	err = cudaMalloc((void **)&d_group_centroids, sizeof(float)*t_groups*logic_dim);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to allocate device vector d_group_centroids (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

    chronometer_t ch_creating_group_filter;
    chrono_reset(&ch_creating_group_filter);
    chrono_start(&ch_creating_group_filter);

	//////////////////////////////////////////
	// CREATING GROUP FILTER USING KMEANS++ //
	//////////////////////////////////////////
	if(verbosity > 2)
		printf("Group filter size = %d\n",t_groups);

	kmeanspp(d_centroids, k, logic_dim, t_groups, 0, d_group_centroids, d_group_filter_labels);
	cudaDeviceSynchronize();
	gpuErrchk( cudaPeekAtLastError() );

	organize_group_filter<<<1,max_threads>>>(d_group_filter_labels, k, t_groups, d_group_filter_locs, d_group_filter_cent_ids);

	cudaDeviceSynchronize();
	gpuErrchk( cudaPeekAtLastError() );
	chrono_stop(&ch_label);
	
	#if DEBUG_KMEANSPP_WRITE_FILES 
		write_data_from_device(
			(int*)d_group_filter_labels,
			k,
			1,
			"./out/group_filter_labels.txt"
		);
		write_data_from_device(
			(int*)d_group_filter_cent_ids,
			k,
			1,
			"./out/group_filter_cent_ids.txt"
		);
		write_data_from_device(
			(int*)d_group_filter_locs,
			t_groups+1,
			1,
			"./out/group_filter_locs.txt"
		);
	#endif



	err = cudaFree(d_group_centroids);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to free device vector d_group_centroids (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	chrono_stop(&ch_creating_group_filter);
	//////////////////////////////////////////





	// #ifdef ORGANIZE_CENTS_IN_MEMORY
	// 	chronometer_t ch_organize_cents;
	// 	chrono_reset(&ch_organize_cents);
	// 	chrono_start(&ch_organize_cents);

	// 	float *d_organized_centroids = NULL; 
	// 	err = cudaMalloc((void **)&d_organized_centroids, sizeof(float)*logic_dim*dataset_size);
	// 	if (err != cudaSuccess){
	// 		fprintf(stderr, "Failed to allocate device vector d_organized_centroids (error code %s)!\n", cudaGetErrorString(err));
	// 		exit(EXIT_FAILURE);
	// 	}

	// 	organize_cents_in_memory<<<nblocks,nthreads>>>(d_centroids, d_organized_centroids, k, logic_dim,
	// 		d_group_filter_labels, 
	// 		d_group_filter_cent_ids, d_group_filter_locs);

	// 	err = cudaFree(d_centroids);
	// 	if (err != cudaSuccess){
	// 		fprintf(stderr, "Failed to free device vector d_dataset (error code %s)!\n", cudaGetErrorString(err));
	// 		exit(EXIT_FAILURE);
	// 	}
	// 	d_centroids = d_organized_centroids;

	// 	chrono_stop(&ch_organize_cents);
	// 	if(verbosity > 2)
	// 		printf("ch_organize_cents time: %.3f milliseconds.\n",((chrono_gettotal(&ch_organize_cents))/1000)/1000.0);
	// #endif
	// if( initialization_method == 0){


	// 	chrono_start(&ch_init_bounds);
		
	// 		cudaMemset(d_labels_change_count, 0, sizeof(uint)*k*nblocks);	
	// 		cudaMemset(d_centroids_change, 0, sizeof(float)*k*logic_dim*nblocks*nwarps);	
	// 		init_bounds_and_assign_labels<<<nblocks,nthreads>>>(
	// 			d_dataset,dataset_size,
	// 			d_centroids,k,
	// 			logic_dim,
	// 			d_labels, d_labels_change_count,
	// 			d_upperbounds,d_lowerbounds,
	// 			t_groups,
	// 			d_group_filter_labels, d_group_lowerbounds,
	// 			d_centroids_change
	// 			,d_group_filter_cent_ids, d_group_filter_locs);
	// 		// );
	// 		cudaDeviceSynchronize();
		
	// 	chrono_stop(&ch_init_bounds);
	// 	gpuErrchk( cudaPeekAtLastError() );


	// 	chrono_start(&ch_first_update);
		
	// 		cudaMemset(d_max_centroid_shift, 0, sizeof(float));	
	// 		cudaMemset(d_reassignments, 0, sizeof(uint));
	// 		cudaMemset(d_max_group_shift, 0, sizeof(float)*t_groups);	
	// 		update<<<k,n_threads_update,sm_size_update>>>(
	// 			d_centroids,
	// 			logic_dim,k,
	// 			d_labels, d_labels_change_count,
	// 			d_labels_count,
	// 			d_new_centroids, d_centroids_change,
	// 			d_centroid_shift,d_max_centroid_shift, 
	// 			d_group_filter_labels,
	// 			d_max_group_shift,
	// 			d_reassignments,
	// 			use_shared_memory_update);
	// 		cudaDeviceSynchronize();

	// 		update_bounds<<<nblocks,nthreads>>>(
	// 			dataset_size,
	// 			logic_dim,k,
	// 			d_labels,
	// 			d_lowerbounds,d_upperbounds,
	// 			d_centroid_shift,d_max_centroid_shift, 
	// 			t_groups,
	// 			d_group_lowerbounds,
	// 			d_max_group_shift);      
	// 		cudaDeviceSynchronize();
	// 	chrono_stop(&ch_first_update);
	// 	gpuErrchk( cudaPeekAtLastError() );

	// 	float* d_old_centroids = d_centroids;
	// 	d_centroids = d_new_centroids;
	// 	d_new_centroids = d_old_centroids;

	// }

    uint n_it = 0;
	float sqrdNormError = MAX_FLOAT;
    while(max_it > n_it){
		n_it++;

        long long time_start_label = chrono_gettotal(&ch_label);
        chrono_start(&ch_label);
		cudaMemset(d_centroids_change, 0, sizeof(float)*k*logic_dim);
		cudaMemset(d_labels_change_count, 0, sizeof(int)*k);

            assign_label<<<nblocks,nthreads>>>(
                d_dataset,dataset_size,
				logic_dim,k,
                d_labels, d_labels_change_count,
                d_centroids, d_centroids_change,
				d_lowerbounds,d_upperbounds,
				t_groups,
				d_group_filter_labels, d_group_lowerbounds,
				d_group_filter_cent_ids, d_group_filter_locs,
				use_shared_memory_assign_label);
            cudaDeviceSynchronize();
        

        chrono_stop(&ch_label);
		gpuErrchk( cudaPeekAtLastError() );


        long long update_start_time = chrono_gettotal(&ch_update);
        chrono_start(&ch_update);

			cudaMemset(d_max_centroid_shift, 0, sizeof(float));	
			cudaMemset(d_reassignments, 0, sizeof(uint));
			cudaMemset(d_max_group_shift, 0, sizeof(float)*t_groups);	
            // update<<<k,n_threads_update>>>(
            //     d_centroids,
			// 	logic_dim,k,
            //     d_labels, d_labels_change_count,
			// 	d_labels_count,
            //     d_new_centroids, d_centroids_change,
			// 	d_centroid_shift,d_max_centroid_shift, 
			// 	d_group_filter_labels,
			// 	d_max_group_shift,
			// 	d_reassignments,
			// 	use_shared_memory_update,
			// 	nblocks, nwarps);

			update_centroids<<<nblocks,nthreads>>>(
				d_centroids,
				logic_dim,k,
				d_labels_change_count,
				d_labels_count,
				d_new_centroids, d_centroids_change);

			update_label_count_and_centroid_shift<<<nblocks,nthreads>>>(
				d_centroids,
				logic_dim,k,
				d_labels_change_count,
				d_labels_count,
				d_new_centroids,
				d_centroid_shift,d_max_centroid_shift, 
				d_group_filter_labels,
				d_max_group_shift,
				d_reassignments);

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


		#if DEBUG_KMEANS_WRITE_FILES 
			char filename[100];
			sprintf(filename,"./out/centroids-it-%03u.txt",n_it);
			write_data_from_device(
				d_centroids,
				k,
				logic_dim,
				filename
			);

			sprintf(filename,"./out/labels-it-%03u.txt",n_it);
			write_data_from_device(
				(int*)d_labels,
				dataset_size,
				1,
				filename
			);

			sprintf(filename,"./out/upperbounds-it-%03u.txt",n_it);
			write_data_from_device(
				d_upperbounds,
				dataset_size,
				1,
				filename
			);
		#endif

    }

	// unalign_memory<<<ceil((float)k/(float)nthreads),nthreads>>>(d_centroids,d_new_centroids, k,dim,logic_dim);
	// cudaDeviceSynchronize();

    //==================
	//  DEALLOC MEMORY
	//==================
    chronometer_t ch_dealloc_mem;
    chrono_reset(&ch_dealloc_mem);
    chrono_start(&ch_dealloc_mem);

	err = cudaFree(d_labels_change_count);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to free device vector d_labels_change_count (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	err = cudaFree(d_labels_count);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to free device vector d_labels_count (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaFree(d_centroids_change);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to free device vector d_centroids_change (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaFree(d_sqrdNormError);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to free device vector d_sqrdNormError (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	if(allocated_upperbounds){
		err = cudaFree(d_upperbounds);
		if (err != cudaSuccess){
			fprintf(stderr, "Failed to free device vector d_upperbounds (error code %s)!\n", cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}
	}

	if(allocated_lowerbounds){
		err = cudaFree(d_lowerbounds);
		if (err != cudaSuccess){
			fprintf(stderr, "Failed to free device vector d_lowerbounds (error code %s)!\n", cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}
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

	err = cudaFree(d_group_filter_cent_ids);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to free device vector d_group_filter_cent_ids (error code %s)!\n", cudaGetErrorString(err));
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



	if(dim != logic_dim){

		d_dataset = d_original_dataset;
		if(allocated_aligned_dataset){
			err = cudaFree(d_aligned_dataset);
			if (err != cudaSuccess){
				fprintf(stderr, "Failed to free device vector d_dataset (error code %s)!\n", cudaGetErrorString(err));
				exit(EXIT_FAILURE);
			}
		}
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


	#if DEBUG_KMEANS_WRITE_FILES 
		write_data_from_device(
			d_centroids,
			k,
			logic_dim,
			"./out/centroids-final.txt"
		);

		write_data_from_device(
			d_dataset,
			dataset_size,
			dim,
			"./out/points.txt"
		);

		write_data_from_device(
			(int*)d_labels,
			dataset_size,
			1,
			"./out/labels-final.txt"
		);

		write_data_from_device(
			d_upperbounds,
			dataset_size,
			1,
			"./out/upperbounds-final.txt"
		);
	#endif

	if(d_output_centroids == NULL){
		err = cudaFree(d_centroids);
		if (err != cudaSuccess){
			fprintf(stderr, "Failed to free device vector d_centroids (error code %s)!\n", cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}

		err = cudaFree(d_new_centroids);
		if (err != cudaSuccess){
			fprintf(stderr, "Failed to free device vector d_new_centroids (error code %s)!\n", cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}
	} else {
		if(d_output_centroids != d_centroids){
			cudaMemcpy(d_output_centroids, d_centroids, sizeof(float)*k*logic_dim, cudaMemcpyDeviceToDevice);

			err = cudaFree(d_centroids);
			if (err != cudaSuccess){
				fprintf(stderr, "Failed to free device vector d_centroids (error code %s)!\n", cudaGetErrorString(err));
				exit(EXIT_FAILURE);
			}
		} else {
			err = cudaFree(d_new_centroids);
			if (err != cudaSuccess){
				fprintf(stderr, "Failed to free device vector d_new_centroids (error code %s)!\n", cudaGetErrorString(err));
				exit(EXIT_FAILURE);
			}
		}

	}

	return;
}