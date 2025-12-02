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
__global__
void label_first_centroid_kmeanspp_logc(float* dataset, uint dataset_size, 
        float* centroids, uint dim, 
        uint* labels, uint* labels_count,
        float* upperbounds, float* lowerbounds,
		uint* candidates, float* max_near_cent_dist,
		uint* n_chosen_centroids
){

    // initialize variables
    uint warpIdx = threadIdx.x / WARP_SIZE;
    uint laneIdx = threadIdx.x % WARP_SIZE;
    uint cent = 0;

    uint nwarps = blockDim.x / WARP_SIZE;

	float max_dist = -1.0; 
	uint candidate_to_centroid = 0;

    for(uint i = warpIdx+blockIdx.x*nwarps; i < dataset_size; i += nwarps*gridDim.x){
		float new_dist = euclidean_distance_sqrd(&dataset[i*dim], &centroids[cent*dim], dim, laneIdx);

        // upperbounds[i] == min dist
        // lowerbounds[i] == sec min dist
		if(laneIdx == 0){
			upperbounds[i] = new_dist;
			lowerbounds[i] = MAX_FLOAT;
			labels[i] = cent;
			if(new_dist > max_dist){
				max_dist = new_dist;
				candidate_to_centroid=i; //i is candidate to next centroid
			}
        }

    }
    if(laneIdx == 0){
        max_near_cent_dist[blockIdx.x*nwarps+warpIdx] = max_dist;
        candidates[blockIdx.x*nwarps+warpIdx] = candidate_to_centroid;
    }
	if(threadIdx.x == 0 && blockIdx.x == 0){
		labels_count[cent]=dataset_size;
		*n_chosen_centroids=1;
	}
}


__global__
void find_new_centroids_kmeanspp_logc(float* dataset, uint dataset_size, 
        float* centroids, 
        uint dim, 
        uint* labels,
		uint* labels_count,
		uint* new_labels_count,
        float* upperbounds, float* lowerbounds,
        int* chosen_centroids,
		uint* n_chosen_centroids,
		uint* candidates, float* max_near_cent_dist,
		uint max_bucket_size
){

    // initialize variables
    uint warpIdx = threadIdx.x / warpSize;
    uint laneIdx = threadIdx.x % warpSize;
    int nwarps = blockDim.x / warpSize;

	int local_n_chosen_centroids = *n_chosen_centroids;

    uint* candidate_to_centroid = &candidates[(blockIdx.x*nwarps+warpIdx)*local_n_chosen_centroids];
    float* max_dist = &max_near_cent_dist[(blockIdx.x*nwarps+warpIdx)*local_n_chosen_centroids];
	for(int c = 0; c < local_n_chosen_centroids; c++){
		candidate_to_centroid[c] = -1;
		max_dist[c] = -1.0f;
	}

    // for(uint i = warpIdx+blockIdx.x*nwarps; i < dataset_size; i += nwarps*blockDim.x){
    for(uint i = warpIdx+blockIdx.x*nwarps; i < dataset_size; i += nwarps*gridDim.x){
		int cent = labels[i];
		if(labels_count[cent] >= max_bucket_size){
			int new_cent = chosen_centroids[cent];

			float new_dist = euclidean_distance_sqrd(&dataset[i*dim], &centroids[new_cent*dim], dim, laneIdx);

			// if(threadIdx.x == 0 && blockIdx.x == 0 ||
			// 	threadIdx.x == 0 && blockIdx.x == 1)
			// 	printf("Block %d: Point %u dist to cent %u = %f dist to new cent %u = %f \n",blockIdx.x,i,cent,upperbounds[i],new_cent,new_dist);

			// upperbounds[i] is the distance from the point 'i' to its nearest centroid
			// lowerbounds[i] The lowerbound is the distance from the point 'i' to its second nearest centroid
			if(laneIdx == 0){
				if(new_dist < lowerbounds[i]){
					if(new_dist < upperbounds[i]){
						lowerbounds[i] = upperbounds[i];
						upperbounds[i] = new_dist;
						labels[i] = new_cent;
						atomicAdd(&new_labels_count[new_cent],1);
						atomicSub(&new_labels_count[cent],1);
						cent=new_cent;
					}
					else
						lowerbounds[i] = new_dist;
				}
			}
		}
		if(laneIdx == 0){
			if(upperbounds[i] > max_dist[cent]){
				max_dist[cent] = upperbounds[i];
				candidate_to_centroid[cent]=i;
			}
		}

    }
       
}


__global__
void append_centroids_kmeanspp_logc(
		float* dataset, uint dataset_size, 
        float* centroids, 
		uint dim,
		uint n_cents_so_far, //number of chosen centroids so far 
		uint* labels_count,
        int* chosen_centroids,
		uint* n_chosen_centroids, //pointer to number of chosen centroids (will be updated)
		uint* finished_centroids, 
        uint* candidates, float* max_min_cent_dist, int total_candidates,
		int max_bucket_size, int max_centroids
){
    __shared__ float sh_max_dist[MAX_THREADS];
    __shared__ uint sh_next_cent[MAX_THREADS];
    
    float max = -1.0;
    uint next_cent=0;
	uint local_finished_centroids = 0;
    for(int j = blockIdx.x; j < n_cents_so_far; j+=gridDim.x){
		if(labels_count[j] < max_bucket_size){
			local_finished_centroids++;
			continue;
		}
		for(int i = threadIdx.x; i < total_candidates; i+=blockDim.x){
			// if(threadIdx.x == 0 && blockIdx.x == 0 || 
			// 	threadIdx.x == 0 && blockIdx.x == 1)
			// 	printf("Block %u Centroid %u: candidate %u with dist %f \n",blockIdx.x,j,candidates[i*n_cents_so_far+j],max_min_cent_dist[i*n_cents_so_far+j]);
			if(max < max_min_cent_dist[i*n_cents_so_far+j]){
				max = max_min_cent_dist[i*n_cents_so_far+j];
				next_cent = candidates[i*n_cents_so_far+j];
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
		// if(sh_max_dist[0] < 0.0){
		// 	// if(threadIdx.x == 0){
		// 	// 	printf("Block %d: No new centroid chosen for centroid %d, label count %u \n",blockIdx.x,j,labels_count[j]);
		// 	// }
		// 	continue; // all centroids are finished
		// }
		uint new_cent = sh_next_cent[0];
		__shared__ uint new_cent_id_sh;
		if(threadIdx.x == 0){
			uint new_cent_id = atomicAdd(n_chosen_centroids,1);
			// if(threadIdx.x == 0)
			// 	printf("Block %d: Chosen new centroid %u with max dist %f at point %u \n",blockIdx.x, new_cent_id,sh_max_dist[0],new_cent);
			if(new_cent_id < max_centroids){
				chosen_centroids[j] = new_cent_id;
				new_cent_id_sh = new_cent_id;
			}
		}
		__syncthreads();
		uint new_cent_id = new_cent_id_sh;
		if(new_cent_id >= max_centroids){
			return; // there is no space for new centroids
		}
		for(int i = threadIdx.x; i < dim; i+=blockDim.x){
			centroids[i+new_cent_id*dim] = dataset[i+new_cent*dim];
		}
	}
	if(threadIdx.x == 0){
		atomicAdd(finished_centroids, local_finished_centroids);
	}
}



__global__
void label_last_centroids_kmeanspp_logc(float* dataset, uint dataset_size, 
        float* centroids, 
        uint dim, 
        uint* labels,
		uint* labels_count,
		uint* new_labels_count,
        float* upperbounds, float* lowerbounds,
        int* chosen_centroids,
		uint max_bucket_size
){

    // initialize variables
    uint warpIdx = threadIdx.x / warpSize;
    uint laneIdx = threadIdx.x % warpSize;
    int nwarps = blockDim.x / warpSize;

    // for(uint i = warpIdx+blockIdx.x*nwarps; i < dataset_size; i += nwarps*blockDim.x){
    for(uint i = warpIdx+blockIdx.x*nwarps; i < dataset_size; i += nwarps*gridDim.x){
		int cent = labels[i];
		if(labels_count[cent] < max_bucket_size)
			continue;
		int new_cent = chosen_centroids[cent];
			
        float new_dist = euclidean_distance_sqrd(&dataset[i*dim], 
				&centroids[new_cent*dim], dim, laneIdx);

        // upperbounds[i] is the distance from the point 'i' to its nearest centroid
        // lowerbounds[i] The lowerbound is the distance from the point 'i' to its second nearest centroid
		if(laneIdx == 0){
			if(new_dist < lowerbounds[i]){
				if(new_dist < upperbounds[i]){
					lowerbounds[i] = upperbounds[i];
					upperbounds[i] = new_dist;
					labels[i] = new_cent;
					atomicAdd(&new_labels_count[new_cent],1);
					atomicSub(&new_labels_count[cent],1);
				}
				else
					lowerbounds[i] = new_dist;
			}
        }

    }
       
}

__global__
void initialize_with_given_cent(
		float* dataset, uint dataset_size, uint dim,
		float* centroids, uint first_centroid_idx
){
	for(uint i = threadIdx.x; i < dim; i+=blockDim.x){
		centroids[i] = dataset[first_centroid_idx*dim + i];
	}
}

int ___kmeanslogc__it = 0;
void kmeanspp_logc(float* d_dataset, uint dataset_size, 
        uint dim, uint max_buckets,
		uint verbosity,
		//OUTPUTS
		float* d_centroids,
        uint* d_labels,
		int* n_buckets_out,
		//OPTIONAL INPUTS
        uint max_bucket_size = 1024, 
		float* d_upperbounds_out = NULL
){
    
    // uint k = max_buckets; //MAX number of centroids to choose
    uint k = max_buckets; //MAX number of centroids to choose

	// ALLOC MEMORY
	cudaError_t err = cudaSuccess;
	int devUsed = 0;
	cudaSetDevice(devUsed);
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, devUsed);

	int max_threads = deviceProp.maxThreadsPerBlock;
	int nthreads = deviceProp.maxThreadsPerMultiProcessor / 2;
	if(nthreads > deviceProp.maxThreadsPerBlock) nthreads = deviceProp.maxThreadsPerBlock;
	int n_mps = deviceProp.multiProcessorCount;
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

	int *d_chosen_centroids = NULL;
	err = cudaMalloc((void **)&d_chosen_centroids, sizeof(uint)*k);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to allocate device vector d_chosen_centroids (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	float *d_max_near_cent_dist = NULL; 
	err = cudaMalloc((void **)&d_max_near_cent_dist, sizeof(float)*nblocks*nwarps*k);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to allocate device vector d_max_near_cent_dist (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	uint *d_candidates_to_nextcent = NULL; 
	err = cudaMalloc((void **)&d_candidates_to_nextcent, sizeof(uint)*nblocks*nwarps*k);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to allocate device vector d_candidates_to_nextcent (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

    uint* d_labels_count = NULL;
    err = cudaMalloc((void **)&d_labels_count, sizeof(uint)*k);
    if (err != cudaSuccess){
        fprintf(stderr, "Failed to allocate device vector d_labels_count (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    } 
    cudaMemset(d_labels_count, 0, sizeof(uint)*k);

	uint* d_new_labels_count = NULL;
	err = cudaMalloc((void **)&d_new_labels_count, sizeof(uint)*k);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to allocate device vector d_new_labels_count (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	uint* d_n_chosen_centroids = NULL;
	err = cudaMalloc((void **)&d_n_chosen_centroids, sizeof(uint));
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to allocate device vector d_n_chosen_centroids (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}


	uint* d_n_finished_centroids = NULL;
	err = cudaMalloc((void **)&d_n_finished_centroids, sizeof(uint));
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to allocate device vector d_n_finished_centroids (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

    //---
	//initialize first centroid at random and set bounds as max float
	setMaxFloat<<<ceil((float)dataset_size/(float)nthreads),nthreads>>>(d_upperbounds,dataset_size);
	setMaxFloat<<<ceil((float)dataset_size/(float)nthreads),nthreads>>>(d_lowerbounds,dataset_size);



	// RANDOM FIRST CENTROID 
	// static unsigned long long seed = time(NULL);
	static unsigned long long seed = 777+___kmeanslogc__it;
	___kmeanslogc__it++;
	int first_cent = rand_r((unsigned int*)&seed) % dataset_size;
	// printf("K-means++ LOGC first centroid index: %d \n",first_cent);
	initialize_with_given_cent<<<1,max_threads>>>(
		d_dataset,dataset_size,dim,
		d_centroids, first_cent);

	cudaDeviceSynchronize();
	gpuErrchk( cudaPeekAtLastError() );


	label_first_centroid_kmeanspp_logc<<<nblocks,nthreads>>>(
		d_dataset,dataset_size,
		d_centroids,
		dim,
		d_labels,
		d_labels_count,
		d_upperbounds,d_lowerbounds,
		d_candidates_to_nextcent, d_max_near_cent_dist,
		d_n_chosen_centroids
	);

	cudaDeviceSynchronize();
	gpuErrchk( cudaPeekAtLastError() );

	uint h_n_chosen_centroids = 1;
	uint h_n_finished_centroids = 0;

	cudaMemset(d_n_finished_centroids , 0, sizeof(uint));
	append_centroids_kmeanspp_logc<<<n_mps,max_threads>>>(
		d_dataset,dataset_size,
		d_centroids,
		dim,
		h_n_chosen_centroids,
		d_labels_count,
		d_chosen_centroids,
		d_n_chosen_centroids,
		d_n_finished_centroids,
		d_candidates_to_nextcent, d_max_near_cent_dist, 
		nblocks*nwarps, max_bucket_size, k
	);
	cudaDeviceSynchronize();
	gpuErrchk( cudaPeekAtLastError() );

	cudaMemcpy(&h_n_chosen_centroids, d_n_chosen_centroids, sizeof(uint), cudaMemcpyDeviceToHost);
	cudaMemcpy(&h_n_finished_centroids, d_n_finished_centroids, sizeof(uint), cudaMemcpyDeviceToHost);


	#define DEBUG_KMEANSPP_LOGC_WRITE_FILES 0
	#if DEBUG_KMEANSPP_LOGC_WRITE_FILES 
		char filename[100];

		sprintf(filename,"./out/points.txt");
		write_data_from_device(
			d_dataset,
			dataset_size,
			dim,
			filename
		);

		sprintf(filename,"./out/centroids-kmeansppLOGC-it-%03u.txt",h_n_chosen_centroids);
		write_data_from_device(
			d_centroids,
			h_n_chosen_centroids,
			dim,
			filename
		);

		sprintf(filename,"./out/labels-kmeansppLOGC-it-%03u.txt",h_n_chosen_centroids);
		write_data_from_device(
			(int*)d_labels,
			dataset_size,
			1,
			filename
		);


		sprintf(filename,"./out/upperbound-kmeansppLOGC-it-%03u.txt",h_n_chosen_centroids);
		write_data_from_device(
			d_upperbounds,
			dataset_size,
			1,
			filename
		);

		sprintf(filename,"./out/labelscount-kmeansppLOGC-it-%03u.txt",h_n_chosen_centroids);
		write_data_from_device(
			(int*)d_labels_count,
			h_n_chosen_centroids,
			1,
			filename
		);

		sprintf(filename,"./out/chosencentroids-kmeansppLOGC-it-%03u.txt",h_n_chosen_centroids);
		write_data_from_device(
			(int*)d_chosen_centroids,
			h_n_chosen_centroids,
			1,
			filename
		);

		sprintf(filename,"./out/candidates-kmeansppLOGC-it-%03u.txt",h_n_chosen_centroids);
		write_data_from_device(
			(int*)d_candidates_to_nextcent,
			nblocks*nwarps*h_n_chosen_centroids,
			1,
			filename
		);
	#endif


	cudaMemcpy(d_new_labels_count, d_labels_count, sizeof(uint)*k, cudaMemcpyDeviceToDevice);
	while(h_n_chosen_centroids < k && h_n_finished_centroids < h_n_chosen_centroids){
		printf("Chosen centroids: %u / %u; Finished centroids: %u \n",h_n_chosen_centroids,k,h_n_finished_centroids);

			


		find_new_centroids_kmeanspp_logc<<<nblocks,nthreads>>>(
			d_dataset,dataset_size,
			d_centroids,
			dim,
			d_labels,
			d_labels_count,
			d_new_labels_count,
			d_upperbounds,d_lowerbounds,
			d_chosen_centroids,
			d_n_chosen_centroids,
			d_candidates_to_nextcent, d_max_near_cent_dist,
			max_bucket_size
		);
		cudaDeviceSynchronize();
		gpuErrchk( cudaPeekAtLastError() );
		cudaMemcpy(d_labels_count, d_new_labels_count, sizeof(uint)*k, cudaMemcpyDeviceToDevice);


		cudaMemset(d_n_finished_centroids , 0, sizeof(uint));
		append_centroids_kmeanspp_logc<<<n_mps,max_threads>>>(
			d_dataset,dataset_size,
			d_centroids,
			dim,
			h_n_chosen_centroids,
			d_labels_count,
			d_chosen_centroids,
			d_n_chosen_centroids,
			d_n_finished_centroids,
			d_candidates_to_nextcent, d_max_near_cent_dist, 
			nblocks*nwarps, max_bucket_size, k
		);
		cudaDeviceSynchronize();
		gpuErrchk( cudaPeekAtLastError() );

		cudaMemcpy(&h_n_chosen_centroids, d_n_chosen_centroids, sizeof(uint), cudaMemcpyDeviceToHost);
		cudaMemcpy(&h_n_finished_centroids, d_n_finished_centroids, sizeof(uint), cudaMemcpyDeviceToHost);

		#if DEBUG_KMEANSPP_LOGC_WRITE_FILES 
			filename[100];
			sprintf(filename,"./out/centroids-kmeansppLOGC-it-%03u.txt",h_n_chosen_centroids);
			write_data_from_device(
				d_centroids,
				h_n_chosen_centroids,
				dim,
				filename
			);

			sprintf(filename,"./out/labels-kmeansppLOGC-it-%03u.txt",h_n_chosen_centroids);
			write_data_from_device(
				(int*)d_labels,
				dataset_size,
				1,
				filename
			);


			sprintf(filename,"./out/upperbound-kmeansppLOGC-it-%03u.txt",h_n_chosen_centroids);
			write_data_from_device(
				d_upperbounds,
				dataset_size,
				1,
				filename
			);

			sprintf(filename,"./out/labelscount-kmeansppLOGC-it-%03u.txt",h_n_chosen_centroids);
			write_data_from_device(
				(int*)d_labels_count,
				h_n_chosen_centroids,
				1,
				filename
			);

			sprintf(filename,"./out/chosencentroids-kmeansppLOGC-it-%03u.txt",h_n_chosen_centroids);
			write_data_from_device(
				(int*)d_chosen_centroids,
				h_n_chosen_centroids,
				1,
				filename
			);

			sprintf(filename,"./out/candidates-kmeansppLOGC-it-%03u.txt",h_n_chosen_centroids);
			write_data_from_device(
				(int*)d_candidates_to_nextcent,
				nblocks*nwarps*h_n_chosen_centroids,
				1,
				filename
			);
		#endif


	}


	printf("K-means++ LOGC chose %u centroids, and %u finished centroids.\n",h_n_chosen_centroids,h_n_finished_centroids);
	
	label_last_centroids_kmeanspp_logc<<<nblocks,nthreads>>>(
		d_dataset,dataset_size,
		d_centroids,
		dim,
		d_labels,
		d_labels_count,
		d_new_labels_count,
		d_upperbounds,d_lowerbounds,
		d_chosen_centroids,
		max_bucket_size
	);

	cudaDeviceSynchronize();
	gpuErrchk( cudaPeekAtLastError() );

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
	err = cudaFree(d_max_near_cent_dist);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to free device vector d_max_near_cent_dist (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	err = cudaFree(d_candidates_to_nextcent);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to free device vector d_candidates_to_nextcent (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
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

	err = cudaFree(d_n_finished_centroids);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to free device vector d_n_finished_centroids (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	*n_buckets_out = h_n_chosen_centroids;
}