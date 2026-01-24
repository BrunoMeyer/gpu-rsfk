#define RANDOM_SEED 123456789

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <float.h>
#include <iostream>
#include <algorithm>
#include <chrono>

#include "kmeans.h"
#include "cpu-utils.c"
#include "defines.h"

#ifdef COMPARE_KMEANS
#include <kmcuda.h>
#endif

#ifdef IMPORT_DATASET
// #include "../import_dataset_8000000_128_2048.h"
#endif

using namespace std;

int main(int argc,char* argv[])
{    
	if(argc < 5){
		// printf("usage: ./kmeans <file-name or '-c'> <dataset_size> <dim> <k>\n");
		// printf("usage: ./kmeans <dataset.txt or '-c'> <dataset_size> <dim> <k> [max_it] [check_method] [tol] [verbosity] [centroids.txt]\n");
		printf("-------------------------\n");
		printf("usage: ./kmeans <dataset.csv or '-c'> <dataset_size> <dim> <k> [max_it] [check_method] [tol] [init_method] [verbosity] [n_execution] [t_groups]\n");
		printf("-------------------------\n");
		printf("check_method: 0 -> until max it\n");
		printf("              1 -> by squared norm error\n");
		printf("              2 -> by number of reassingments (default)\n");
		printf("-------------------------\n");
		printf("init_method:  0 -> random\n");
		printf("              1 -> kmeans++ (default)\n");
		printf("-------------------------\n");
		return 0;
	}

	srand(RANDOM_SEED);


	// uint dataset_tests[] = {}
	//=========================
	//       PARAMETERS
	//=========================
	uint dataset_size = atoi(argv[2]);//1000000;
	uint dim = atoi(argv[3]); // dim < 129
	int k = atoi(argv[4]); 
	uint max_it = 100;
	uint check_method = 2;
	float tolerance = 0.00;
	uint verbosity = 2;
	uint init = 1;
	uint n_execution = 0;
	// uint t_groups = ceil(k/10);
	uint t_groups = 64;
	if(cmdOptionExists(argv, argc+argv, "-old")){ //must be the last parameter
		if(argc > 6)
			max_it = atoi(argv[5]);
		if(argc > 7)
			check_method = atoi(argv[6]); 
		if(argc > 8) 
			tolerance = atof(argv[7]); 
		if(argc > 9) 
			init = atoi(argv[8]); 
		if(argc > 10) 
			verbosity = atoi(argv[9]); 
		if(argc > 11) 
			n_execution = atoi(argv[10]); 
	} else {
			max_it = getIntFromCmdOption(argv, argc+argv, "-maxiter", 100);
			check_method = getIntFromCmdOption(argv, argc+argv, "-check", 2);
			tolerance = getIntFromCmdOption(argv, argc+argv, "-tol", 0.00);
			init = getIntFromCmdOption(argv, argc+argv, "-init", 1);
			verbosity = getIntFromCmdOption(argv, argc+argv, "-verb", 2);
			n_execution = getIntFromCmdOption(argv, argc+argv, "-nexec", 1)-1;

	}
	// if(argc > 11) 
	// 	t_groups = atoi(argv[11]); 
	//=========================

	uint* labels = (uint*) malloc(sizeof(uint)*dataset_size);
	float* centroids = (float*) malloc(sizeof(float)*k*dim);
	float* dataset;
	if(cmdOptionExists(argv, argc+argv, "-p")){
		int total_elements;
		int truedim;
		read_csv(argv[1],&total_elements,&truedim,&dataset);
		if(truedim != dim){
			printf("error: wrong dimension %d != %d\n",truedim,dim);
			return -1;
		}
		// shuffle_data(dataset, total_elements, dim, dataset_size );
		//todo shuffle function
	} else {
		dataset = (float*) malloc(sizeof(float)*dataset_size*dim);
		if(argv[1][0] == '-' && argv[1][1] == 'c'){
			create_data(dataset,dataset_size*dim);
		}
		else{
			// char file_name[120];
			from_file(argv[1],dataset_size,dim,dataset);
		}
	}


	// //==================== 
	// //    data set size benchmark
	// //==================== 
	// #define BENCHMARK
	// uint dataset_tests[] = {125000,250000,500000,1000000,2000000,4000000,8000000,16000000};
	// uint n_dset_test = 8;
	// for(uint i = 0; i < n_dset_test; i++){
	// 	uint dataset_size=dataset_tests[i];
	// //==================== 
	// //    k benchmark
	// //==================== 
	// #define BENCHMARK
	// uint k_tests[] = {32,64,128,256,512,1024,2048,4096};
	// uint n_k_test = 8;
	// for(uint i = 0; i < n_k_test; i++){
	// 	uint k = k_tests[i];
	// //==================== 
	// //    t benchmark
	// //====================
	// #define BENCHMARK
	// const uint n_t_test = 1;//6;
	// uint t_tests[n_t_test] = {1};//,16,32,64,128,256,512};
	// for(uint i = 0; i < n_t_test; i++){
	// 	uint t_groups = t_tests[i];
	// //====================
	printf("dataset size = %i dim = %i K = %i \n",dataset_size, dim, k); 
	printf("max_it = %d check_method = %d tolerance = %.3f init = %d\n",max_it, check_method, tolerance, init);
	printf("t_groups=%d verbosity = %d n_execution = %d\n",t_groups, verbosity, n_execution);
	auto t1_timetmp = std::chrono::high_resolution_clock::now();

		kmeansGpu(dataset, dataset_size, dim, k, max_it, check_method, tolerance, init, t_groups, verbosity, labels, centroids);

	auto t2_timetmp = std::chrono::high_resolution_clock::now();
	float total_time = std::chrono::duration_cast<std::chrono::microseconds>(t2_timetmp - t1_timetmp).count();

	if(n_execution) for(uint i = 0; i < n_execution; i++){
		auto t1_timetmp = std::chrono::high_resolution_clock::now();

			kmeansGpu(dataset, dataset_size, dim, k, max_it, check_method, tolerance, init, t_groups, 1, labels, centroids);

		auto t2_timetmp = std::chrono::high_resolution_clock::now();
		float total_time = std::chrono::duration_cast<std::chrono::microseconds>(t2_timetmp - t1_timetmp).count();

		#ifdef COMPARE_KMEANS
			printf("---> total_time kmeans-michel: %.3f miliseconds\n",(float)total_time/1000.0);
			float average_distance;
			auto t1_kmcuda = std::chrono::high_resolution_clock::now();

				KMCUDAResult result = kmeans_cuda(
					kmcudaInitMethodPlusPlus, NULL,  // kmeans++ centroids initialization
					// kmcudaInitMethodRandom, NULL,  // random initialization
					0.01,                            // less than 1% of the dataset are reassigned in the end
					0.1,                             // activate Yinyang refinement with 0.1 threshold
					kmcudaDistanceMetricL2,          // Euclidean distance
					dataset_size, dim, k,
					0xDEADBEEF,                      // random generator seed
					0,                               // use all available CUDA devices
					-1,                              // dataset are supplied from host
					0,                               // not in float16x2 mode
					1,                               // moderate verbosity
					dataset, centroids, labels, &average_distance);

			auto t2_kmcuda = std::chrono::high_resolution_clock::now();
			float total_timekmcuda = std::chrono::duration_cast<std::chrono::microseconds>(t2_kmcuda - t1_kmcuda).count();
			printf("---> total_time kmeans-Yinyang: %.3f miliseconds\n",(float)total_timekmcuda/1000.0);
		#endif
	} else 
		write_data("centroids.txt",k,dim,centroids);
	printf("\n");
	// #ifdef BENCHMARK
	// 	}
	// #endif


	free(dataset);
	free(labels);
	free(centroids);

}