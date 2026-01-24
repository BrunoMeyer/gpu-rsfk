#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <float.h>

// #include "kmeans.h"
#include "defines.h"

#include "kernel-functions/gpu-utils.cu"

#include "kernel-functions/kmeanspp.cu"


//FIRST ITERATION 
template<bool USE_FLOAT4=false>
__global__
void label_first_centroid_kmeans_hierarchical(
		float* dataset, int dataset_size, 
        float* centroids, int dim, 
        int* labels, 
		int* labels_count, int* new_labels_count,
		int first_centroid,
        float* upperbounds, float* lowerbounds,
		int* next_centroids,
		int* chosen_points,
		int* n_chosen_centroids, //pointer to number of chosen centroids (will be updated)
		int* not_finished_centroids,
		int random_number
){

    // initialize variables
    int warpIdx = threadIdx.x / WARP_SIZE;
    int laneIdx = threadIdx.x % WARP_SIZE;
    int cent = 0;

    int nwarps = blockDim.x / WARP_SIZE;

    for(int i = warpIdx+blockIdx.x*nwarps; i < dataset_size; i += nwarps*gridDim.x){
		float new_dist;
		if constexpr (USE_FLOAT4){
			// aligned data
			new_dist = warp_euclidean_distance_sqrd_float4(&dataset[i*dim], &centroids[cent*dim], dim, laneIdx);
		}
		else{
			// unaligned data
			new_dist = warp_euclidean_distance_sqrd(&dataset[i*dim], &centroids[cent*dim], dim, laneIdx);
		}

        // upperbounds[i] == min dist
        // lowerbounds[i] == sec min dist
		if(laneIdx == 0){
			upperbounds[i] = new_dist;
			lowerbounds[i] = MAX_FLOAT;
			labels[i] = cent;
        }
    }
	if(threadIdx.x == 0 && blockIdx.x == 0){
		int point =  random_number % dataset_size;
		next_centroids[0] = point;
		chosen_points[0] = first_centroid;
		labels_count[0] = dataset_size;
		new_labels_count[0] = dataset_size;
		*n_chosen_centroids = 1;
		*not_finished_centroids = 0;
	}
}

template<bool USE_FLOAT4=false>
__global__
void label_hierarchical_kmeans(
		float* dataset, int dataset_size, 
        float* centroids, 
        int dim, 
        int* labels,
		int* labels_count,
		int* new_labels_count,
        float* upperbounds, float* lowerbounds,
        int* new_centroids,
		int max_bucket_size
){

    // initialize variables
    int warpIdx = threadIdx.x / warpSize;
    int laneIdx = threadIdx.x % warpSize;
    int nwarps = blockDim.x / warpSize;
	int wid_grid = warpIdx + blockIdx.x * nwarps;
	
	int next_cent_count = 0;
	int next_cent_pos = -1;
	int probe_centroid_idx = -1;

    for(int i = wid_grid; i < dataset_size; i += nwarps*gridDim.x){
		int cent = labels[i];
		if(labels_count[cent] >= max_bucket_size){
			int new_cent = new_centroids[cent];
			if(new_cent != -1){
				float new_dist;
				if constexpr (USE_FLOAT4){
					// aligned data
					new_dist = warp_euclidean_distance_sqrd_float4(&dataset[i*dim], &centroids[new_cent*dim], dim, laneIdx);
				}
				else{
					// unaligned data
					new_dist = warp_euclidean_distance_sqrd(&dataset[i*dim], &centroids[new_cent*dim], dim, laneIdx);
				}
				if(new_dist < lowerbounds[i]){
					if(laneIdx == 0){
						if(new_dist < upperbounds[i]){
							lowerbounds[i] = upperbounds[i];
							upperbounds[i] = new_dist;
							labels[i] = new_cent;

							atomicAdd(&new_labels_count[new_cent],1);
							atomicSub(&new_labels_count[cent],1);
							cent = new_cent;
						}
						else
							lowerbounds[i] = new_dist;
					}
				}
			}
		}
    }
}

__global__
void find_next_centroids_hierarchical_kmeans(
		float* dataset, int dataset_size, 
        int* labels,
		int* labels_count,
		int* next_centroids, 
		int* chosen_points,
		int* n_chosen_centroids,
		int max_bucket_size,
		int* not_finished_centroids,
		int n_probes,
		int random_number
){
	extern __shared__ int probe_cent_count_sm[];
	int probe_offset = n_probes*blockIdx.x;
	int* new_cent_pos_sm = &probe_cent_count_sm[n_probes];

	for(int i = threadIdx.x; i < n_probes; i += blockDim.x){
		probe_cent_count_sm[i] = 0;
		new_cent_pos_sm[i] = -1;
		if(i + probe_offset < n_chosen_centroids[0]){
			int label_count = labels_count[i + probe_offset];
			if(label_count >= max_bucket_size)
				new_cent_pos_sm[i] = random_number % (label_count - 1);
		}
	}

    for(int i = threadIdx.x; i < dataset_size; i += blockDim.x){
		int cent = labels[i];
		int probe_centroid_idx = cent - probe_offset;
		if(probe_centroid_idx < 0 || probe_centroid_idx >= n_probes)
			continue; //not a centroid probed by this block
		int pos = atomicAdd(&probe_cent_count_sm[probe_centroid_idx],1);
		if(pos == new_cent_pos_sm[probe_centroid_idx]){
			if(i == chosen_points[probe_centroid_idx]){
				//do not choose the same point again
				next_centroids[probe_centroid_idx] = -1; 
			} else {
				next_centroids[probe_centroid_idx] = i;
			}
		}
    }
	if(threadIdx.x == 0 && blockIdx.x == 0){
		*not_finished_centroids = 0;
	}
}


__global__
void append_centroids_hierarchical_kmeans(
		float* dataset, int dataset_size, 
        float* centroids, 
		int dim,
		int n_cents_so_far, //number of chosen centroids so far 
		int* labels_count,
		int* new_labels_count,
        int* new_centroids,
		int* next_centroids,
		int* chosen_points,
		int* n_chosen_centroids, //pointer to number of chosen centroids (will be updated)
		int* not_finished_centroids, 
		int max_bucket_size, int max_centroids
){
	__shared__ int new_cent_pos_sh;
	int not_finished_local = 0;
	for(int i = blockIdx.x; i < n_cents_so_far; i+=gridDim.x){
		if(new_labels_count[i] >= max_bucket_size){
			not_finished_local++;
			int new_cent_from_point = next_centroids[i];
			if(new_cent_from_point == -1){
				//no new centroid was found for this centroid
				new_centroids[i] = -1;
			} else {
				//append new centroid
				if(threadIdx.x == 0){
					new_cent_pos_sh = atomicAdd(n_chosen_centroids,1);
				}
				__syncthreads();
				int new_cent_pos = new_cent_pos_sh;
				if(new_cent_pos < max_centroids){
					for(int j = threadIdx.x; j < dim; j+=blockDim.x){
						centroids[j+new_cent_pos*dim] = dataset[j+new_cent_from_point*dim];
					}
					if(threadIdx.x == 0){
						new_centroids[i] = new_cent_pos;
						labels_count[new_cent_pos] = 0;
						new_labels_count[new_cent_pos] = 0;
						chosen_points[new_cent_pos] = new_cent_from_point;
						next_centroids[i] = -1; //reset next centroid for this centroid
					}
				} else {
					new_centroids[i] = -1; //indicate there is no space for new centroids
				}
			}
		} else {
			if(threadIdx.x == 0){
				new_centroids[i] = -1;
			}
		}
		if(threadIdx.x == 0){
			atomicAdd(not_finished_centroids, not_finished_local);
			labels_count[i] = new_labels_count[i];
		}
	}
}


template<bool USE_FLOAT4=false>
__global__
void label_new_centroids_hierarchical_kmeans(float* dataset, int dataset_size, 
        float* centroids, 
        int dim, 
        int* labels,
		int* labels_count,
        float* upperbounds, float* lowerbounds,
        int* new_centroids
){

    // initialize variables
    int warpIdx = threadIdx.x / warpSize;
    int laneIdx = threadIdx.x % warpSize;
    int nwarps = blockDim.x / warpSize;

    // for(int i = warpIdx+blockIdx.x*nwarps; i < dataset_size; i += nwarps*blockDim.x){
    for(int i = warpIdx+blockIdx.x*nwarps; i < dataset_size; i += nwarps*gridDim.x){
		int cent = labels[i];
		int new_cent = new_centroids[cent];
		if(new_cent == -1)
			continue; //no new centroid assigned to this centroid
		// if(new_cent > n_max_centroids)
		// 	if(laneIdx == 0)
		// 		printf("Error: point %u has label %d greater than n_max_centroids %d \n",i,new_cent,n_max_centroids);
			
        // float new_dist = warp_euclidean_distance_sqrd(&dataset[i*dim], 
				// &centroids[new_cent*dim], dim, laneIdx);
		float new_dist;
		if constexpr (USE_FLOAT4){
			// aligned data
			new_dist = warp_euclidean_distance_sqrd_float4(&dataset[i*dim], &centroids[new_cent*dim], dim, laneIdx);
		}
		else{
			// unaligned data
			new_dist = warp_euclidean_distance_sqrd(&dataset[i*dim], &centroids[new_cent*dim], dim, laneIdx);
		}


        // upperbounds[i] is the distance from the point 'i' to its nearest centroid
        // lowerbounds[i] The lowerbound is the distance from the point 'i' to its second nearest centroid
		if(new_dist < lowerbounds[i]){
			if(laneIdx == 0){
				if(new_dist < upperbounds[i]){
					lowerbounds[i] = upperbounds[i];
					upperbounds[i] = new_dist;
					labels[i] = new_cent;
					atomicAdd(&labels_count[new_cent],1);
					atomicSub(&labels_count[cent],1);
				}
				else
					lowerbounds[i] = new_dist;
			}
        }

    }
       
}

template<bool USE_FLOAT4=false> //default unaligned data (it can be used with both aligned and unaligned data)
void hierarchical_kmeans(float* d_dataset, int dataset_size, 
        int dim, int max_buckets,
		int verbosity,
		//OUTPUTS
		float* d_centroids,
        int* d_labels,
		int* n_buckets_out,
		//OPTIONAL INPUTS
        int max_bucket_size = 256, 
		float* d_upperbounds_out = NULL
){
	printf("Starting Hierarchical K-means with max %u buckets and max bucket size %u...\n",max_buckets, max_bucket_size);
	// ALLOC MEMORY
	cudaError_t err = cudaSuccess;
	int devUsed = 0;
	cudaSetDevice(devUsed);
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, devUsed);

	int max_threads = deviceProp.maxThreadsPerBlock;
	int nthreads = deviceProp.maxThreadsPerMultiProcessor / 2;
	if(nthreads > deviceProp.maxThreadsPerBlock) nthreads = deviceProp.maxThreadsPerBlock;
	int nblocks = deviceProp.multiProcessorCount*(deviceProp.maxThreadsPerMultiProcessor/nthreads);
	int max_shared_mem = deviceProp.sharedMemPerBlock;

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

	int *d_next_centroids = NULL;
	err = cudaMalloc((void **)&d_next_centroids, sizeof(int)*max_buckets);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to allocate device vector d_next_centroids (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	int *d_new_centroids = NULL;
	err = cudaMalloc((void **)&d_new_centroids, sizeof(int)*max_buckets);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to allocate device vector d_new_centroids (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

    int* d_labels_count = NULL;
    err = cudaMalloc((void **)&d_labels_count, sizeof(int)*max_buckets);
    if (err != cudaSuccess){
        fprintf(stderr, "Failed to allocate device vector d_labels_count (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    } 

	int* d_new_labels_count = NULL;
	err = cudaMalloc((void **)&d_new_labels_count, sizeof(int)*max_buckets);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to allocate device vector d_new_labels_count (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	int* d_n_chosen_centroids = NULL;
	err = cudaMalloc((void **)&d_n_chosen_centroids, sizeof(int));
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to allocate device vector d_n_chosen_centroids (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}


	int* d_n_not_finished_centroids = NULL;
	err = cudaMalloc((void **)&d_n_not_finished_centroids, sizeof(int));
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to allocate device vector d_n_not_finished_centroids (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	int* d_chosen_points = NULL;
	err = cudaMalloc((void **)&d_chosen_points, sizeof(int)*max_buckets);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to allocate device vector d_chosen_points (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

    //---
	//initialize first centroid at random and set bounds as max float
	setMaxFloat<<<ceil((float)dataset_size/(float)nthreads),nthreads>>>(d_upperbounds,dataset_size);
	setMaxFloat<<<ceil((float)dataset_size/(float)nthreads),nthreads>>>(d_lowerbounds,dataset_size);



	// RANDOM FIRST CENTROID 
	static unsigned long long seed = time(NULL);
	int first_cent = rand_r((unsigned int*)&seed) % dataset_size;
	// printf("K-means++ LOGC first centroid index: %d \n",first_cent);
	initialize_with_given_cent<<<1,max_threads>>>(
		d_dataset,dataset_size,dim,
		d_centroids, first_cent);

	cudaDeviceSynchronize();
	gpuErrchk( cudaPeekAtLastError() );

	int random_number = rand_r((unsigned int*)&seed);
	label_first_centroid_kmeans_hierarchical<USE_FLOAT4><<<nblocks,nthreads>>>(
		d_dataset,dataset_size,
		d_centroids,
		dim,
		d_labels,
		d_labels_count, d_new_labels_count,
		first_cent,
		d_upperbounds,d_lowerbounds,
		d_next_centroids,
		d_chosen_points,
		d_n_chosen_centroids,
		d_n_not_finished_centroids,
		random_number
	);

	cudaDeviceSynchronize();
	gpuErrchk( cudaPeekAtLastError() );


	int h_n_chosen_centroids = 1;
	int h_n_not_finished_centroids = 1;


	#define DEBUG_HIERARCHICAL_KMEANS_WRITE_FILES 1
	#if DEBUG_HIERARCHICAL_KMEANS_WRITE_FILES 
		char filename[100];

		sprintf(filename,"./out/labelscount-hierarchical-kmeans-it-%03u.txt",h_n_chosen_centroids);
		write_data_from_device(
			(int*)d_new_labels_count,
			h_n_chosen_centroids,
			1,
			filename
		);

		sprintf(filename,"./out/chosenpoints-hierarchical-kmeans-it-%03u.txt",h_n_chosen_centroids);
		write_data_from_device(
			(int*)d_new_centroids,
			h_n_chosen_centroids,
			1,
			filename
		);

		#if DEBUG_HIERARCHICAL_KMEANS_WRITE_FILES > 1

			sprintf(filename,"./out/next_centroids-hierarchical-kmeans-it-%03u.txt",h_n_chosen_centroids);
			write_data_from_device(
				(int*)d_next_centroids,
				h_n_chosen_centroids,
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

			sprintf(filename,"./out/centroids-hierarchical-kmeans-it-%03u.txt",h_n_chosen_centroids);
			write_data_from_device(
				d_centroids,
				h_n_chosen_centroids,
				dim,
				filename
			);
	
			sprintf(filename,"./out/labels-hierarchical-kmeans-it-%03u.txt",h_n_chosen_centroids);
			write_data_from_device(
				(int*)d_labels,
				dataset_size,
				1,
				filename
			);
	
			sprintf(filename,"./out/upperbound-hierarchical-kmeans-it-%03u.txt",h_n_chosen_centroids);
			write_data_from_device(
				d_upperbounds,
				dataset_size,
				1,
				filename
			);

			sprintf(filename,"./out/new_centroids-hierarchical-kmeans-it-%03u.txt",h_n_chosen_centroids);
			write_data_from_device(
				(int*)d_new_centroids,
				h_n_chosen_centroids,
				1,
				filename
			);
		
		#endif

	#endif

	append_centroids_hierarchical_kmeans<<<nblocks,nthreads>>>(
		d_dataset,dataset_size,
		d_centroids,
		dim,
		1, //number of chosen centroids so far 
		d_labels_count,
		d_new_labels_count,
		d_new_centroids,
		d_next_centroids,
		d_chosen_points,
		d_n_chosen_centroids,
		d_n_not_finished_centroids,
		max_bucket_size, max_buckets
	);


	cudaMemcpy(&h_n_chosen_centroids, d_n_chosen_centroids, sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(&h_n_not_finished_centroids, d_n_not_finished_centroids, sizeof(int), cudaMemcpyDeviceToHost);


	printf("Hierarchical K-means iteration: chosen centroids %u, not_finished centroids %u \n",h_n_chosen_centroids,h_n_not_finished_centroids);
	while(h_n_chosen_centroids < max_buckets && h_n_not_finished_centroids > 0){

		label_hierarchical_kmeans<USE_FLOAT4><<<nblocks,nthreads>>>(
			d_dataset,dataset_size,
			d_centroids,
			dim,
			d_labels,
			d_labels_count,
			d_new_labels_count,
			d_upperbounds,d_lowerbounds,
			d_new_centroids,
			max_bucket_size
		);
		cudaDeviceSynchronize();
		gpuErrchk( cudaPeekAtLastError() );


		random_number = rand_r((unsigned int*)&seed);
		int nprobes = (max_shared_mem/sizeof(int))/2;
		int shared_mem_size = nprobes*2*sizeof(int);
		int nblocks_probes = ceil((float)h_n_chosen_centroids/(float)nprobes);
		find_next_centroids_hierarchical_kmeans<<<nblocks_probes,max_threads,shared_mem_size>>>(
			d_dataset,dataset_size,
			d_labels,
			d_new_labels_count,
			d_next_centroids,
			d_chosen_points,
			d_n_chosen_centroids,
			max_bucket_size,
			d_n_not_finished_centroids,
			nprobes,
			random_number
		);

		cudaDeviceSynchronize();
		gpuErrchk( cudaPeekAtLastError() );

		#if DEBUG_HIERARCHICAL_KMEANS_WRITE_FILES 
			printf("Writing debug files for hierarchical k-means iteration with %u chosen centroids...\n",h_n_chosen_centroids);

			#if DEBUG_HIERARCHICAL_KMEANS_WRITE_FILES > 1

				sprintf(filename,"./out/centroids-hierarchical-kmeans-it-%03u.txt",h_n_chosen_centroids);
				write_data_from_device(
					d_centroids,
					h_n_chosen_centroids,
					dim,
					filename
				);
	
				sprintf(filename,"./out/labels-hierarchical-kmeans-it-%03u.txt",h_n_chosen_centroids);
				write_data_from_device(
					(int*)d_labels,
					dataset_size,
					1,
					filename
				);
	

				sprintf(filename,"./out/upperbound-hierarchical-kmeans-it-%03u.txt",h_n_chosen_centroids);
				write_data_from_device(
					d_upperbounds,
					dataset_size,
					1,
					filename
				); 
				
			#endif

			sprintf(filename,"./out/labelscount-hierarchical-kmeans-it-%03u.txt",h_n_chosen_centroids);
			write_data_from_device(
				(int*)d_new_labels_count,
				h_n_chosen_centroids,
				1,
				filename
			);

			sprintf(filename,"./out/chosenpoints-hierarchical-kmeans-it-%03u.txt",h_n_chosen_centroids);
			write_data_from_device(
				(int*)d_chosen_points,
				h_n_chosen_centroids,
				1,
				filename
			);

			sprintf(filename,"./out/next_centroids-hierarchical-kmeans-it-%03u.txt",h_n_chosen_centroids);
			write_data_from_device(
				(int*)d_next_centroids,
				h_n_chosen_centroids,
				1,
				filename
			);

			sprintf(filename,"./out/new_centroids-hierarchical-kmeans-it-%03u.txt",h_n_chosen_centroids);
			write_data_from_device(
				(int*)d_new_centroids,
				h_n_chosen_centroids,
				1,
				filename
			);
		#endif



		append_centroids_hierarchical_kmeans<<<nblocks,nthreads>>>(
			d_dataset,dataset_size,
			d_centroids,
			dim,
			h_n_chosen_centroids, //number of chosen centroids so far 
			d_labels_count,
			d_new_labels_count,
			d_new_centroids,
			d_next_centroids,
			d_chosen_points,
			d_n_chosen_centroids,
			d_n_not_finished_centroids,
			max_bucket_size, max_buckets
		);
		cudaDeviceSynchronize();
		gpuErrchk( cudaPeekAtLastError() );

		cudaMemcpy(&h_n_chosen_centroids, d_n_chosen_centroids, sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(&h_n_not_finished_centroids, d_n_not_finished_centroids, sizeof(int), cudaMemcpyDeviceToHost);


		printf("Hierarchical K-means iteration: chosen centroids %u, not_finished centroids %u \n",h_n_chosen_centroids,h_n_not_finished_centroids);


	}


	// printf("K-means++ LOGC chose %u centroids, and %u not_finished centroids.\n",h_n_chosen_centroids,h_n_not_finished_centroids);
	printf("Hierarchical K-means chose %u centroids.\n",h_n_chosen_centroids);

	// FINAL LABELING WITH THE CHOSEN CENTROIDS
	label_new_centroids_hierarchical_kmeans<USE_FLOAT4><<<nblocks,nthreads>>>(
		d_dataset,dataset_size,
		d_centroids,
		dim,
		d_labels,
		d_labels_count,
		d_upperbounds,d_lowerbounds,
		d_new_centroids
	);

	cudaDeviceSynchronize();
	gpuErrchk( cudaPeekAtLastError() );

	#if DEBUG_HIERARCHICAL_KMEANS_WRITE_FILES 
		printf("Writing debug files for hierarchical k-means iteration with %u chosen centroids...\n",h_n_chosen_centroids);

		#if DEBUG_HIERARCHICAL_KMEANS_WRITE_FILES > 1

			sprintf(filename,"./out/centroids-hierarchical-kmeans-it-%03u.txt",h_n_chosen_centroids);
			write_data_from_device(
				d_centroids,
				h_n_chosen_centroids,
				dim,
				filename
			);

			sprintf(filename,"./out/labels-hierarchical-kmeans-it-%03u.txt",h_n_chosen_centroids);
			write_data_from_device(
				(int*)d_labels,
				dataset_size,
				1,
				filename
			);


			sprintf(filename,"./out/upperbound-hierarchical-kmeans-it-%03u.txt",h_n_chosen_centroids);
			write_data_from_device(
				d_upperbounds,
				dataset_size,
				1,
				filename
			);

		#endif

		sprintf(filename,"./out/labelscount-hierarchical-kmeans-it-%03u.txt",h_n_chosen_centroids);
		write_data_from_device(
			(int*)d_labels_count,
			h_n_chosen_centroids,
			1,
			filename
		);

		sprintf(filename,"./out/chosenpoints-hierarchical-kmeans-it-%03u.txt",h_n_chosen_centroids);
		write_data_from_device(
			(int*)d_chosen_points,
			h_n_chosen_centroids,
			1,
			filename
		);

		sprintf(filename,"./out/next_centroids-hierarchical-kmeans-it-%03u.txt",h_n_chosen_centroids);
		write_data_from_device(
			(int*)d_next_centroids,
			h_n_chosen_centroids,
			1,
			filename
		);

		sprintf(filename,"./out/new_centroids-hierarchical-kmeans-it-%03u.txt",h_n_chosen_centroids);
		write_data_from_device(
			(int*)d_new_centroids,
			h_n_chosen_centroids,
			1,
			filename
		);
		exit(EXIT_FAILURE);

	#endif

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
    err = cudaFree(d_labels_count);
    if (err != cudaSuccess){
        fprintf(stderr, "Failed to free device vector d_labels_count (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

	err = cudaFree(d_new_labels_count);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to free device vector d_new_labels_count (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaFree(d_n_chosen_centroids);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to free device vector d_n_chosen_centroids (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaFree(d_n_not_finished_centroids);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to free device vector d_n_not_finished_centroids (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	*n_buckets_out = h_n_chosen_centroids;

	err = cudaFree(d_next_centroids);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to free device vector d_next_centroids (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	err = cudaFree(d_new_centroids);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to free device vector d_new_centroids (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

}