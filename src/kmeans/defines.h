
#ifndef DEFINES_KMEANSGPU
    #define DEFINES_KMEANSGPU

    #define REPORT_ALL

    #define MAX_SHARED_MEMORY_IN_BYTES 102400
    #define MAX_SM_PER_MP 102400
    #define MAX_SM_PER_BLOCK 49152

    #define READ_CENTROIDS

    // #define MAX_DIM 120
    #define N_MP 56

    // tava:
    // #define BLOCKS_PER_MP 3
    // #define N_THREADS 512

    //ficou:
    // #define BLOCKS_PER_MP 2
    // #define N_THREADS 768

    //ficou:
    #define BLOCKS_PER_MP 1
    #define N_THREADS 1024
    // #define N_THREADS (768+128)

    #define N_BLOCKS (N_MP*BLOCKS_PER_MP)
    #define MAX_THREADS 1024
    // #define N_BLOCKS 56
    #define NTA (N_BLOCKS*N_THREADS)

    #define WARP_SIZE 32
    #define N_WARPS (N_THREADS/WARP_SIZE)


    #define MAX_FLOAT 340282346638528859811704183484516925440.0000000000000000
    #define FULL 0xffffffff

    #define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
    inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true){
        if (code != cudaSuccess){
            fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
            if (abort) exit(code);
        }
    }
#endif