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


#endif