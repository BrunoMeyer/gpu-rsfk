all: main_archall python_archall

main_archall: src/rpfk.cu
	nvcc -O3 -gencode=arch=compute_30,code=sm_30 -gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_75,code=sm_75 -std=c++11 -I/usr/include/python2.7/include,/usr/include/python2.7 -lpython2.7 src/rpfk.cu -o rpfk

python_archall: src/python/pymodule_ext.cu
	nvcc -O3 -gencode=arch=compute_30,code=sm_30 -gencode=arch=compute_75,code=sm_75 -std=c++11 -I/usr/include/python2.7/include,/usr/include/python2.7 -lpython2.7 --compiler-options '-fPIC' -o python/gpu_rpfk/librpfk.so --shared src/python/pymodule_ext.cu

main_arch30: src/rpfk.cu
	nvcc -O3 -gencode=arch=compute_30,code=sm_30 -std=c++11 -I/usr/include/python2.7/include,/usr/include/python2.7 -lpython2.7 src/rpfk.cu -o rpfk

python_arch30: src/python/pymodule_ext.cu
	nvcc -O3 -gencode=arch=compute_30,code=sm_30 -std=c++11 -I/usr/include/python2.7/include,/usr/include/python2.7 -lpython2.7 --compiler-options '-fPIC' -o python/gpu_rpfk/librpfk.so --shared src/python/pymodule_ext.cu

main_arch61: src/rpfk.cu
	nvcc -O3 -gencode=arch=compute_61,code=sm_61 -std=c++11 -I/usr/include/python2.7/include,/usr/include/python2.7 -lpython2.7 src/rpfk.cu -o rpfk

python_arch61: src/python/pymodule_ext.cu
	nvcc -O3 -gencode=arch=compute_61,code=sm_61 -std=c++11 -I/usr/include/python2.7/include,/usr/include/python2.7/include,/usr/include/python2.7 -lpython2.7 --compiler-options '-fPIC' -o python/gpu_rpfk/librpfk.so --shared src/python/pymodule_ext.cu

main_arch75: src/rpfk.cu
	nvcc -O3 -gencode=arch=compute_75,code=sm_75 -std=c++11 -I/usr/include/python2.7/include,/usr/include/python2.7 -lpython2.7 src/rpfk.cu -o rpfk

python_arch75: src/python/pymodule_ext.cu
	nvcc -O3 -gencode=arch=compute_75,code=sm_75 -std=c++11 -I/usr/include/python2.7/include,/usr/include/python2.7/include,/usr/include/python2.7 -lpython2.7 --compiler-options '-fPIC' -o python/gpu_rpfk/librpfk.so --shared src/python/pymodule_ext.cu


clean:
	rm -f rpfk
	rm -f python/gpu_rpfk/librpfk.so