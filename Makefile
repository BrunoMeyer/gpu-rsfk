all: main python

main: rptk.cu
	nvcc -O3 -gencode=arch=compute_30,code=sm_30 -std=c++11 -I/usr/include/python2.7 -lpython2.7 rptk.cu -o rptk

python: pymodule_ext.cu
	nvcc -O3 -gencode=arch=compute_30,code=sm_30 -std=c++11 -I/usr/include/python2.7 -lpython2.7 --compiler-options '-fPIC' -o librptk.so --shared pymodule_ext.cu
