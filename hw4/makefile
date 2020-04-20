# basic simple Makefile starter
#

all:
	nvcc -Xcompiler -fopenmp -O2 -o matvec-gpu matvec-gpu.cu
	nvcc -Xcompiler -fopenmp -O2 -o jacobi2D-gpu jacobi2D-gpu.cu

clean:
	rm matvec-gpu
	rm jacobi2D-gpu
