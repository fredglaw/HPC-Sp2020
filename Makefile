# basic simple Makefile starter
#
CFLAGS= -Wall -g

all:
	g++ -std=c++11 -g -o val_test01_solved val_test01_solved.cpp
	g++ -std=c++11 -g -o val_test02_solved val_test02_solved.cpp
	g++ -std=c++11 -fopenmp -march=native -O2 -o MMult1 MMult1.cpp
	g++ -std=c++11 -fopenmp -g -o omp_solved2 omp_solved2.c
	g++ -std=c++11 -fopenmp -g -o omp_solved3 omp_solved3.c
	g++ -std=c++11 -fopenmp -g -o omp_solved4 omp_solved4.c
	g++ -std=c++11 -fopenmp -g -o omp_solved5 omp_solved5.c
	g++ -std=c++11 -fopenmp -g -o omp_solved6 omp_solved6.c
	g++ -std=c++11 -fopenmp -O2 -o jacobi2D-omp jacobi2D-omp.cpp
	g++ -std=c++11 -fopenmp -O2 -o gs2D-omp gs2D-omp.cpp

clean:
	rm val_test01_solved
	rm val_test02_solved
	rm omp_solved2
	rm omp_solved3
	rm omp_solved4
	rm omp_solved5
	rm omp_solved6
	rm MMult1
	rm jacobi2D-omp
	rm gs2D-omp
