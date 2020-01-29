all main.cu:
	nvcc -gencode arch=compute_30,code=sm_30 -std=c++11 -I/usr/include/python2.7 -lpython2.7 main.cu
