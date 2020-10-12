all: main_archall python_archall

main_archall: src/rpfk.cu
	nvcc -O3 -gencode=arch=compute_30,code=sm_30 -gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_75,code=sm_75 -std=c++11 src/rpfk.cu -o rpfk

python_archall: src/python/pymodule_ext.cu
	nvcc -O3 -gencode=arch=compute_30,code=sm_30 -gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_75,code=sm_75 -std=c++11 --compiler-options '-fPIC' -o python/gpu_rpfk/librpfk.so --shared src/python/pymodule_ext.cu

main_arch30: src/rpfk.cu
	nvcc -O3 -gencode=arch=compute_30,code=sm_30 -std=c++11 src/rpfk.cu -o rpfk

python_arch30: src/python/pymodule_ext.cu
	nvcc -O3 -gencode=arch=compute_30,code=sm_30 -std=c++11 --compiler-options '-fPIC' -o python/gpu_rpfk/librpfk.so --shared src/python/pymodule_ext.cu

main_arch61: src/rpfk.cu
	nvcc -O3 -gencode=arch=compute_61,code=sm_61 -std=c++11 src/rpfk.cu -o rpfk

python_arch61: src/python/pymodule_ext.cu
	nvcc -O3 -gencode=arch=compute_61,code=sm_61 -std=c++11 --compiler-options '-fPIC' -o python/gpu_rpfk/librpfk.so --shared src/python/pymodule_ext.cu

main_arch75: src/rpfk.cu
	nvcc -lnvgraph -O3 -gencode=arch=compute_75,code=sm_75 -std=c++11 src/rpfk.cu -o rpfk

python_arch75: src/python/pymodule_ext.cu
	nvcc -lnvgraph -O3 -gencode=arch=compute_75,code=sm_75 -std=c++11 --compiler-options '-fPIC' -o python/gpu_rpfk/librpfk.so --shared src/python/pymodule_ext.cu

install:
	rm -rf python/build python/dist python/*.eg-info
	cd python; python3 setup.py install $(SETUP_FLAG)
clean:
	rm -f rpfk
	rm -f python/gpu_rpfk/librpfk.so
	rm -rf python/build/
	rm -rf python/dist/
	rm -rf python/gpu_rpfk.egg-info/
