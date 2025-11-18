#ifndef KMEANSGPU_H

    #define KMEANSGPU_H

    #include "./kmeans.cu"

    void kmeansGpu(float* d_dataset, uint dataset_size, 
        uint dim, uint k, 
		uint max_it, 
		uint check_method, float tolerance, 
		uint initialization_method, //0 - random, 1 - from cpu
		uint t_groups,
		uint verbosity,
        uint* d_labels, float* d_centroids // <-- outputs
    );

#endif