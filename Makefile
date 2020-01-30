all main.cu:
	nvcc -O3 --gpu-architecture=compute_30 --gpu-code=sm_30 -std=c++11 -I/usr/include/python2.7 -lpython2.7 main.cu
