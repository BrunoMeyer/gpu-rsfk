#ifndef GPUUTILS

#define GPUUTILS

__device__
static inline
float euclidean_distance_sqr(
		float* p0,
		float* p1,
		uint dim,
		uint lane
){
	float4 a,b;
	float s = 0.0f;

	for(uint i=lane; i < dim/4; i+=WARP_SIZE){
		a = reinterpret_cast<float4*>(p0)[i];
		b = reinterpret_cast<float4*>(p1)[i];
		float diff = a.x - b.x;
		s+=diff*diff;
		diff = a.y - b.y;
		s+=diff*diff;
		diff = a.z - b.z;
		s+=diff*diff;
		diff = a.w - b.w;
		s+=diff*diff;
	}
	s += __shfl_xor_sync( 0xffffffff, s,  1); // assuming warpSize=32
	s += __shfl_xor_sync( 0xffffffff, s,  2); // assuming warpSize=32
	s += __shfl_xor_sync( 0xffffffff, s,  4); // assuming warpSize=32
	s += __shfl_xor_sync( 0xffffffff, s,  8); // assuming warpSize=32
	s += __shfl_xor_sync( 0xffffffff, s, 16); // assuming warpSize=32		
	
	// all lanes have the value, just return it
	return s;
}


__device__
static inline
float euclidean_distance_sqrd(
		float* p0,
		float* p1,
		uint dim,
		uint lane
){
	float4 a,b;
	float s = 0.0f;

	for(uint i=lane; i < dim; i+=WARP_SIZE){
        float diff = p0[i] - p1[i];
        s+=diff*diff;
	}
	s += __shfl_xor_sync( 0xffffffff, s,  1); // assuming warpSize=32
	s += __shfl_xor_sync( 0xffffffff, s,  2); // assuming warpSize=32
	s += __shfl_xor_sync( 0xffffffff, s,  4); // assuming warpSize=32
	s += __shfl_xor_sync( 0xffffffff, s,  8); // assuming warpSize=32
	s += __shfl_xor_sync( 0xffffffff, s, 16); // assuming warpSize=32		
	
	// all lanes have the value, just return it
	return s;
}

__device__
inline
void add_points(
		float* a,
		float* b,
		uint dim,
		uint lane
){
	for(uint i=lane; i < dim; i+=WARP_SIZE){
		a[i] = a[i] + b[i];
	}
}


__device__ static float atomicMaxFloat(float* address, float val){
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
            __float_as_int(::fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

__global__
void setMaxFloat(float* mem, uint size){
    uint i = blockIdx.x*blockDim.x+threadIdx.x;
    if(i < size)
        mem[i]=MAX_FLOAT;
}

template <typename T>
__inline__ __device__
void warp_find_max(
        const T* __restrict__ arr,
        int K,
		int lane,
        T &max_val,
        int &max_pos)
{

    T local_val = -FLT_MAX;
    int local_pos = -1;

	// 1) Each thread finds its local max
    for(int j = lane; j < K; j += 32){   // warpSize=32
        T v = arr[j];
        if (v > local_val){
            local_val = v;
            local_pos = j;
        }
    }

    // 2) Warp reduction
    for (int offset = 16; offset > 0; offset /= 2){
        T v = __shfl_down_sync(0xffffffff, local_val, offset);
        int p = __shfl_down_sync(0xffffffff, local_pos, offset);

        if (v > local_val){
            local_val = v;
            local_pos = p;
        }
    }

    // return the result
    max_val = local_val;
    max_pos = local_pos;
}

template <typename T>
__inline__ __device__
T warp_find_max(
        const T* __restrict__ arr,
        int K,
		int lane)
{

    T local_val = -FLT_MAX;
 
	// 1) Each thread finds its local max
    for(int j = lane; j < K; j += 32){   // warpSize=32
        T v = arr[j];
        if (v > local_val){
            local_val = v;
        }
    }

    // 2) Warp reduction
    for (int offset = 16; offset > 0; offset /= 2){
        T v = __shfl_down_sync(0xffffffff, local_val, offset);
 
        if (v > local_val){
            local_val = v;
        }
    }

    // return the result
	return local_val;
}

template <typename T>
__inline__ __device__
T warp_reduction(
        const T local_val)
{
    for (int offset = 16; offset > 0; offset /= 2){
        T v = __shfl_down_sync(0xffffffff, local_val, offset);
 
        if (v > local_val){
            local_val = v;
        }
    }

    // return the result
	return local_val;
}
#endif