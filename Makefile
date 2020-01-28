all main.cu:
	nvcc -std=c++11 -I/usr/include/python2.7 -lpython2.7 main.cu
