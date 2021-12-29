all: main_archall python_archall

main_archall: src/rsfk.cu
	nvcc -lnvgraph -O3 -gencode=arch=compute_30,code=sm_30 -gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_75,code=sm_75 -std=c++11 --compiler-options '-fPIC' src/rsfk.cu -o rsfk

python_archall: src/python/pymodule_ext.cu
	nvcc -O3 -gencode=arch=compute_30,code=sm_30 -gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_75,code=sm_75 -std=c++11 --compiler-options '-fPIC' -o python/gpu_rsfk/librsfk.so --shared src/python/pymodule_ext.cu

main_arch30: src/rsfk.cu
	nvcc -O3 -gencode=arch=compute_30,code=sm_30 -std=c++11 src/rsfk.cu -o rsfk

python_arch30: src/python/pymodule_ext.cu
	nvcc -O3 -gencode=arch=compute_30,code=sm_30 -std=c++11 --compiler-options '-fPIC' -o python/gpu_rsfk/librsfk.so --shared src/python/pymodule_ext.cu

main_arch50: src/rsfk.cu
	nvcc -O3 -gencode=arch=compute_50,code=sm_50 -std=c++11 src/rsfk.cu -o rsfk

python_arch50: src/python/pymodule_ext.cu
	nvcc -O3 -gencode=arch=compute_50,code=sm_50 -std=c++11 --compiler-options '-fPIC' -o python/gpu_rsfk/librsfk.so --shared src/python/pymodule_ext.cu

main_arch61: src/rsfk.cu
	nvcc -O3 -gencode=arch=compute_61,code=sm_61 -std=c++11 src/rsfk.cu -o rsfk

python_arch61: src/python/pymodule_ext.cu
	nvcc -O3 -gencode=arch=compute_61,code=sm_61 -std=c++11 --compiler-options '-fPIC' -o python/gpu_rsfk/librsfk.so --shared src/python/pymodule_ext.cu

main_arch75: src/rsfk.cu
	nvcc -lnvgraph -O3 -gencode=arch=compute_75,code=sm_75 -std=c++11 src/rsfk.cu -o rsfk

python_arch75: src/python/pymodule_ext.cu
	nvcc -lnvgraph -O3 -gencode=arch=compute_75,code=sm_75 -std=c++11 --compiler-options '-fPIC' -o python/gpu_rsfk/librsfk.so --shared src/python/pymodule_ext.cu

main_arch35: src/rsfk.cu
	nvcc -std=c++14 -lnvgraph -O3 -gencode=arch=compute_35,code=sm_35 -std=c++11 src/rsfk.cu -o rsfk

python_arch35: src/python/pymodule_ext.cu
	nvcc -std=c++14 -lnvgraph -O3 -gencode=arch=compute_35,code=sm_35 -std=c++11 --compiler-options '-fPIC' -o python/gpu_rsfk/librsfk.so --shared src/python/pymodule_ext.cu

main_arch86: src/rsfk.cu
	nvcc -lnvgraph -O3 -gencode=arch=compute_86,code=sm_86 -std=c++11 src/rsfk.cu -o rsfk

python_arch86: src/python/pymodule_ext.cu
	nvcc -std=c++14 -lnvgraph -O3 -gencode=arch=compute_86,code=sm_86 -std=c++11 --compiler-options '-fPIC' -o python/gpu_rsfk/librsfk.so --shared src/python/pymodule_ext.cu

install:
	rm -rf python/build python/dist python/*.eg-info
	cd python; python3 setup.py install $(SETUP_FLAG)
clean:
	rm -f rsfk
	rm -f python/gpu_rsfk/librsfk.so
	rm -rf python/build/
	rm -rf python/dist/
	rm -rf python/gpu_rsfk.egg-info/
