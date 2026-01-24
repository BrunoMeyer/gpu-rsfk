#ifndef GPUUTILS

#define GPUUTILS

__device__
static __forceinline__
float warp_euclidean_distance_sqrd_float4(
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

__device__ static __forceinline__
float warp_sqrt_coop(float x) {
    // Initial guess using hardware rsqrt + one Newton step (very good)
    float y = __frsqrt_rn(x);           // reciprocal sqrt, fast hardware approx
    y = __fmaf_rn(y, 0.5f, y);          // better guess: y = 1.5f * y
    y = __fmaf_rn(-x * y * y, 0.5f * y, y);  // one Newton step on 1/sqrt

    // Now do 2–3 cooperative Newton-Raphson steps across the warp
    for (int i = 0; i < 3; ++i) {
        float v = __shfl_xor_sync(0xffffffff, y, 16);
        y = __fmaf_rn(-x * y * v, y, y);   // y ← y * (3 - x*y*v)/2  (fused)
        v = __shfl_xor_sync(0xffffffff, y, 8);
        y = __fmaf_rn(-x * y * v, y, y);
        v = __shfl_xor_sync(0xffffffff, y, 4);
        y = __fmaf_rn(-x * y * v, y, y);
        v = __shfl_xor_sync(0xffffffff, y, 2);
        y = __fmaf_rn(-x * y * v, y, y);
        v = __shfl_xor_sync(0xffffffff, y, 1);
        y = __fmaf_rn(-x * y * v, y, y);
    }
    return __fmaf_rn(y, x, 0.0f);  // y * x → final sqrt(x)
}

__device__ float simple_fast_warp_sqrt(float x) {
    float y = __frsqrt_rn(x);                 // hardware reciprocal sqrt
    y = y * (1.5f - 0.5f * x * y * y);        // 1 Newton step (on 1/sqrt(x))
    y = y * (1.5f - 0.5f * x * y * y);        // 2nd Newton step → very accurate

    // One cooperative refinement step (optional but helps on older arch)
    float t = __shfl_xor_sync(0xffffffff, y, 16);
    y = fmaf(-x * y * t, y, y);               // equivalent to y*(3 - x*y*t)/2
    return x * y;
}

__device__
static __forceinline__
float warp_euclidean_distance_float4(
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
	
	#define SQRT_KMEANS_METHOD 1
	#if SQRT_KMEANS_METHOD == 1
		// 1) all lanes calculate sqrt
		return __fsqrt_rn(s);
	#elif SQRT_KMEANS_METHOD == 2
		// 2) only lane 0 calculates sqrt, then broadcast
		if(lane==0)
			s = __fsqrt_rn(s);
		s = __shfl_sync(0xffffffff, s, 0); // broadcast s from lane 0 to all lanes
		return s;
	#elif SQRT_KMEANS_METHOD == 21
		// 2.1) all lanes calculate sqrt, then lane 0 broadcasts (just for testing)
		// if(lane==0)
			s = __fsqrt_rn(s);
		s = __shfl_sync(0xffffffff, s, 0); // broadcast s from lane 0 to all lanes
		return s;
	#elif SQRT_KMEANS_METHOD == 3
		// 3) warp fast sqrt (does not work)
		if (lane == 0) {
			float test = __frsqrt_rn(s);
			float y = __frsqrt_rn(s);
			y = fmaf(y, fmaf(y, -s*y, 3.0f), 0.0f) * 0.5f;  // one Newton
			s = s * y;
			printf("warp_euclidean_distance_float4: lane %d fast sqrt=%f fsqrt=%f\n", lane, s, test); // there is something wrong, the results are too different
		}
		return __shfl_sync(0xffffffff, s, 0);
	#elif SQRT_KMEANS_METHOD == 4
		// 4) warp cooperative sqrt
		// return warp_sqrt_coop(s);
		float result = simple_fast_warp_sqrt(s);
		// if (lane==0){
		// 	float test = __frsqrt_rn(s);
		// 	float y = __frsqrt_rn(s);
		// 	y = fmaf(y, fmaf(y, -s*y, 3.0f), 0.0f) * 0.5f;  // one Newton
		// 	s = s * y;

		// 	printf("warp_euclidean_distance_float4: lane %d coop sqrt=%f fast sqrt (method 3)=%f fsqrt=%f\n", lane, result, s, test);
		// }
		return result;
	#else
		return __fsqrt_rn(s);
	#endif
}

__device__
static __forceinline__
float warp_euclidean_distance_sqrd(
		float* p0,
		float* p1,
		uint dim,
		uint lane
){
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
static __forceinline__
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
static
__forceinline__ __device__
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
__forceinline__ __device__
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
__forceinline__ __device__
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