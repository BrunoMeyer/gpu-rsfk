all: main_archall python_archall

main_archall: rptk.cu
	nvcc -O3 -gencode=arch=compute_30,code=sm_30 -gencode=arch=compute_75,code=sm_75 -std=c++11 -I/usr/include/python2.7 -lpython2.7 rptk.cu -o rptk

python_archall: pymodule_ext.cu
	nvcc -O3 -gencode=arch=compute_30,code=sm_30 -gencode=arch=compute_75,code=sm_75 -std=c++11 -I/usr/include/python2.7 -lpython2.7 --compiler-options '-fPIC' -o librptk.so --shared pymodule_ext.cu

main_arch30: rptk.cu
	nvcc -O3 -gencode=arch=compute_30,code=sm_30 -std=c++11 -I/usr/include/python2.7 -lpython2.7 rptk.cu -o rptk

python_arch30: pymodule_ext.cu
	nvcc -O3 -gencode=arch=compute_30,code=sm_30 -std=c++11 -I/usr/include/python2.7 -lpython2.7 --compiler-options '-fPIC' -o librptk.so --shared pymodule_ext.cu

main_arch75: rptk.cu
	nvcc -O3 -gencode=arch=compute_75,code=sm_75 -std=c++11 -I/usr/include/python2.7 -lpython2.7 rptk.cu -o rptk

python_arch75: pymodule_ext.cu
	nvcc -O3 -gencode=arch=compute_75,code=sm_75 -std=c++11 -I/usr/include/python2.7 -lpython2.7 --compiler-options '-fPIC' -o librptk.so --shared pymodule_ext.cu

clean:
	rm -f rptk
	rm -f librptk.so