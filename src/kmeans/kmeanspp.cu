#ifndef KMEANSPP_CU
#define KMEANSPP_CU

#include "kernel-functions/gpu-utils.cu"
#include "kernel-functions/kmeanspp_kernels.cu"

/////////////////////////
//      K-MEANS++      //
/////////////////////////
// Note: kmeanspp does not need to take sqrd distances into account, 
// as it only uses distances to choose centroids.
template <bool INDIRECT_POINTS = false>
static inline
void kmeanspp_workflow_async(
	float* d_dataset, uint dataset_size, 
	uint dim, uint k,
	//OUTPUTS
	float* d_centroids,
	uint* d_labels,
	float* d_upperbounds,
	float* d_lowerbounds,
	uint* d_chosen_centroids,
	uint* d_candidates_to_nextcent,
	float* d_max_min_cent_dist,
	//KERNEL PARAMS
	int max_threads,
	int nthreads,
	int nblocks,
	//OPTIONAL
	int* points_indexes = NULL,
    cudaStream_t stream = 0
){
	int nwarps = nthreads / WARP_SIZE;
    //WARNING: max_threads has to be power of 2
    if((max_threads & (max_threads - 1)) != 0)
        printf("WARNING: max_threads has to be power of 2\n");


	//---
	//initialize first centroid at random and set bounds as max float
	setMaxFloat<<<ceil((float)dataset_size/(float)nthreads),nthreads,0,stream>>>(d_upperbounds,dataset_size);
	setMaxFloat<<<ceil((float)dataset_size/(float)nthreads),nthreads,0,stream>>>(d_lowerbounds,dataset_size);
	
	// RANDOM FIRST CENTROID 
	static unsigned long long seed = time(NULL);
	// initialize_first_cent_kmeanspp<<<1,max_threads>>>(
	// 	d_dataset,dataset_size,dim,
	// 	d_centroids, d_chosen_centroids, seed);
	// seed+=1;

	int first_cent = rand_r((unsigned int*)&seed) % dataset_size;
	// printf("K-means++ LOGC first centroid index: %d \n",first_cent);
	initialize_with_given_cent<INDIRECT_POINTS><<<1,max_threads,0,stream>>>(
		d_dataset,dataset_size,dim,
		d_centroids, first_cent, points_indexes);


	// GET FIRST POINT TO BE THE FIRST CENTROID
	// initialize_first_cent_kmeanspp<<<1,max_threads>>>(
	// 	d_dataset,dataset_size,logic_dim,
	// 	d_centroids, d_chosen_centroids);
	
	// cudaDeviceSynchronize();
	// gpuErrchk( cudaPeekAtLastError() );

	for(uint n_chosen_centroids = 1; n_chosen_centroids < k; n_chosen_centroids++){
		find_new_centroid_kmeanspp<false, INDIRECT_POINTS><<<nblocks,nthreads,0,stream>>>(
			d_dataset,dataset_size,
			d_centroids,n_chosen_centroids,
			dim,
			d_labels,
			d_upperbounds,d_lowerbounds,
			d_chosen_centroids,
			d_candidates_to_nextcent, d_max_min_cent_dist,
			points_indexes
		);
		// cudaDeviceSynchronize();
		// gpuErrchk( cudaPeekAtLastError() );

        int shmem_append_centroids = max_threads*sizeof(int)+max_threads*sizeof(float) ;
		append_centroid_kmeanspp<INDIRECT_POINTS><<<1,max_threads,shmem_append_centroids,stream>>>(
			d_dataset,dataset_size,
			d_centroids,n_chosen_centroids,
			dim,
			d_chosen_centroids,
			d_candidates_to_nextcent, d_max_min_cent_dist, nblocks*nwarps,
			points_indexes
		);
		// cudaDeviceSynchronize();
		// gpuErrchk( cudaPeekAtLastError() );

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
		sprintf(filename,"./out/kmeanspp-chosen_centroids.txt");
		write_data_from_device(
			(int*)d_chosen_centroids,
			k,
			1,
			filename
		);

		sprintf(filename,"./out/points.txt");
		write_data_from_device(
			d_dataset,
			dataset_size,
			dim,
			filename
		);
	#endif
	label_last_centroid_kmeanspp<false, INDIRECT_POINTS><<<nblocks,nthreads,0,stream>>>(
		d_dataset,dataset_size,
		d_centroids,k,
		dim,
		d_labels,
		d_upperbounds,d_lowerbounds,
		points_indexes
	);

	// cudaDeviceSynchronize();
	// gpuErrchk( cudaPeekAtLastError() );
}

template <bool INDIRECT_POINTS = false>
static inline
void kmeanspp_workflow(
	float* d_dataset, uint dataset_size, 
	uint dim, uint k, 
	//OUTPUTS
	float* d_centroids,
	uint* d_labels,
	float* d_upperbounds,
	float* d_lowerbounds,
	uint* d_chosen_centroids,
	uint* d_candidates_to_nextcent,
	float* d_max_min_cent_dist,
	//KERNEL PARAMS
	int max_threads,
	int nthreads,
	int nblocks,
	//OPTIONAL
	int* points_indexes = NULL
){
	kmeanspp_workflow_async<INDIRECT_POINTS>(
		d_dataset, dataset_size,
		dim, k,
		//OUTPUTS
		d_centroids,
		d_labels,
		d_upperbounds,
		d_lowerbounds,
		d_chosen_centroids,
		d_candidates_to_nextcent,
		d_max_min_cent_dist,
		//KERNEL PARAMS
		max_threads,
		nthreads,
		nblocks,
		//OPTIONAL
		points_indexes
	);
	cudaDeviceSynchronize();
	gpuErrchk( cudaPeekAtLastError() );
}


template <bool INDIRECT_POINTS = false>
void kmeanspp(float* d_dataset, uint dataset_size, 
        uint dim, uint k, 
		uint verbosity,
		//OUTPUTS
		float* d_centroids,
        uint* d_labels,
		float* d_upperbounds_out = NULL,
		int* points_indexes = NULL
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
	if(d_upperbounds_out != NULL){
		d_upperbounds = d_upperbounds_out;
	}
	else{
		err = cudaMalloc((void **)&d_upperbounds, sizeof(float)*dataset_size);
		if (err != cudaSuccess){
			fprintf(stderr, "Failed to allocate device vector d_upperbounds (error code %s)!\n", cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}
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

	kmeanspp_workflow<INDIRECT_POINTS>(
		d_dataset, dataset_size,
		dim, k,
		//OUTPUTS
		d_centroids,
		d_labels,
		d_upperbounds,
		d_lowerbounds,
		d_chosen_centroids,
		d_candidates_to_nextcent,
		d_max_min_cent_dist,
		//KERNEL PARAMS
		max_threads,
		nthreads,
		nblocks,
		//OPTIONAL
		points_indexes
	);

	err = cudaFree(d_lowerbounds);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to free device vector d_lowerbounds (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	if(d_upperbounds_out == NULL){
		err = cudaFree(d_upperbounds);
		if (err != cudaSuccess){
			fprintf(stderr, "Failed to free device vector d_upperbounds (error code %s)!\n", cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}
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
	static unsigned long long seed = time(NULL);
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
		find_new_centroid_kmeanspp<true><<<nblocks,nthreads>>>(
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
	
	label_last_centroid_kmeanspp<true><<<nblocks,nthreads>>>(
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

#endif