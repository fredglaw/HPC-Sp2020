// script to compute vec-vecs and mat-vecs on gpu. compare to cpu for reference
// g++-9 -std=c++11 -fopenmp -march=native -O2 -o matvec-gpu matvec-gpu.cpp && ./matvec-gpu
// nvcc -Xcompiler -fopenmp -O2 -o matvec-gpu matvec-gpu.cu && ./matvec-gpu
#include <stdio.h>
#include <math.h>
#include <omp.h>

#define N_THREADS 1
#define BLOCK_SIZE 1024

//parallel vecvec to run on the cpu.
void vecvec(double* c, double* x, double* y, long N){
  double inner_prod = 0;
  #pragma omp parallel for reduction(+:inner_prod) num_threads(N_THREADS)
  for (long i=0; i<N; i++){
    inner_prod += x[i]*y[i];
  }
  c[0] = inner_prod;
}

//vevec kernel for gpu
__global__ void vecvec_kernel(double* sum, const double* a, const double* b, long N){
  __shared__ double smem[BLOCK_SIZE];
  int idx = (blockIdx.x) * blockDim.x + threadIdx.x;

  // each thread reads data from global into shared memory
  if (idx < N) smem[threadIdx.x] = a[idx]*b[idx];
  else smem[threadIdx.x] = 0;
  __syncthreads();

  // x >>= 1 means "set x to itself shifted by one bit to the right", i.e., a divison by 2
  // write to memory with threadIdx rather than ``index''
  for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
	if (threadIdx.x < s) {
		smem[threadIdx.x] += smem[threadIdx.x + s];
	}
	__syncthreads();
   }

  // write to global memory
  if (threadIdx.x == 0) sum[blockIdx.x] = smem[threadIdx.x];
}

//reduction kernel
__global__ void reduction_kernel2(double* sum, const double* a, long N){
  __shared__ double smem[BLOCK_SIZE];
  int idx = (blockIdx.x) * blockDim.x + threadIdx.x;

  // each thread reads data from global into shared memory
  if (idx < N) smem[threadIdx.x] = a[idx];
  else smem[threadIdx.x] = 0;
  __syncthreads();

  // x >>= 1 means "set x to itself shifted by one bit to the right", i.e., a divison by 2
  // write to memory with threadIdx rather than ``index''
  for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
	if (threadIdx.x < s) {
		smem[threadIdx.x] += smem[threadIdx.x + s];
	}
	__syncthreads();
   }

  // write to global memory
  if (threadIdx.x == 0) sum[blockIdx.x] = smem[threadIdx.x];
}



void matvec(long m, long n, double *a, double *b, double *c) {
  // does matrix vector multiplication a*b=c, here a is a matrix of size m by n.
  // assume a is stored ROW-MAJOR ORDER
  for (long i=0; i<m;i++) vecvec(c+i,a+(i*n),b,n); //just do vecvec per row,
  //but moving the pointer, this assumes a is row major ordered
}


//CUDA error check
void Check_CUDA_Error(const char *message){
  cudaError_t error = cudaGetLastError();
  if(error!=cudaSuccess) {
    fprintf(stderr,"ERROR: %s: %s\n", message, cudaGetErrorString(error) );
    exit(-1);
  }
}


__global__ void basicwrite(double* addr1, double* addr2){
  //just writes from address1 to address2
  addr2[0]=addr1[0];
}


int main(int argc,char *argv[]){

  long N = (1UL<<24);
  long N_REPEATS = 10;

  //vec-vec setup
  double sum, sum_ref;
  double* x = (double*) malloc(N*sizeof(double)); //allocate x
  double* y = (double*) malloc(N*sizeof(double)); //allocate y
  //#pragma omp parallel for num_threads(N_THREADS) //initialize
  for (long k=0;k<N;k++){
    x[k] = k;
    y[k] = 1.;
  }


  printf("vec-vec size N is: %d\n",N);

  //compute vec-vec reference solution
  double tcpu = omp_get_wtime();
  for (long counter=0; counter<N_REPEATS; counter++){
    vecvec(&sum_ref,x,y,N);
  }
  printf("vec-vec CPU Bandwidth = %f GB/s\n", 2*N*sizeof(double) / (omp_get_wtime()-tcpu)/(N_REPEATS*1e9));

  //cudaMalloc space
  double *x_d, *y_d, *buff_d; //declare pointers for memcopy to device
  cudaMalloc(&x_d, N*sizeof(double));
  cudaMalloc(&y_d, N*sizeof(double));
  long N_work = 1;
  for (long i = (N+BLOCK_SIZE-1)/(BLOCK_SIZE); i > 1; i = (i+BLOCK_SIZE-1)/(BLOCK_SIZE)) N_work += i;
  cudaMalloc(&buff_d, N_work*sizeof(double)); // extra memory buffer for reduction across thread-blocks

  double tgpu = omp_get_wtime();
  for (long counter=0; counter<N_REPEATS; counter++){
    cudaMemcpyAsync(x_d,x,N*sizeof(double),cudaMemcpyHostToDevice); //H2D
    cudaMemcpyAsync(y_d,y,N*sizeof(double),cudaMemcpyHostToDevice); //H2D
    cudaDeviceSynchronize();
    double* sum_d = buff_d;
    long Nb = (N+BLOCK_SIZE-1)/(BLOCK_SIZE);
    vecvec_kernel<<<Nb,BLOCK_SIZE>>>(sum_d, x_d, y_d, N);
    while (Nb > 1) {
      long Nd = Nb;
      Nb = (Nb+BLOCK_SIZE-1)/(BLOCK_SIZE);
      reduction_kernel2<<<Nb,BLOCK_SIZE>>>(sum_d + Nd, sum_d, Nd);
      sum_d += Nd;
    }
    cudaMemcpyAsync(&sum,sum_d, 1*sizeof(double), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize(); //synchronize blocks, i.e. wait for kernel
  }
  printf("vec-vec GPU Bandwidth = %f GB/s\n", 2*N*sizeof(double) / (omp_get_wtime()-tgpu)/(N_REPEATS*1e9));

  //vec-vec error
  double err = abs(sum_ref - sum);
  printf("vec-vec error: %1.2E\n",err);

  //clean up
  cudaFree(x_d);
  cudaFree(y_d);
  cudaFree(buff_d);

  free(x);
  free(y);


/* *************************************************************************
****************************************************************************
********************************  Mat-vec  *********************************
****************************************************************************
**************************************************************************** */

 N = (1UL<<10);
 long M = (1UL<<16);
 //N = 2;
 //long M = 2;
 double* a = (double*) malloc(M*N*sizeof(double));
 double* b = (double*) malloc(N*sizeof(double));
 double* c_ref = (double*) malloc(M*sizeof(double));
 double* c = (double*) malloc(M*sizeof(double));

 for (long i=0; i<M*N; i++) a[i] = 1./(i+1);
 for (long i=0; i<N; i++) b[i] = 1./(2*i+1);
 for (long i=0; i<M; i++){
   c_ref[i] = 0;
   c[i] = 0;
 }

 printf("************************************************\n");
 printf("mat-vec, matrix size is M=%d by N=%d\n",M,N);

 //compute mat-vec reference solution
 tcpu = omp_get_wtime();
 for (long counter=0; counter<N_REPEATS; counter++){
   matvec(M,N,a,b,c_ref);
 }
 printf("mat-vec CPU Bandwidth = %f GB/s\n", 2*M*N*sizeof(double) / (omp_get_wtime()-tcpu)/(N_REPEATS*1e9));

 //cudaMalloc space
 double *a_d, *b_d, *c_d; //declare pointers for memcopy to device
 cudaMalloc(&a_d, M*N*sizeof(double));
 cudaMalloc(&b_d, N*sizeof(double));
 cudaMalloc(&c_d, M*sizeof(double));
 N_work = 1;
 for (long i = (N+BLOCK_SIZE-1)/(BLOCK_SIZE); i > 1; i = (i+BLOCK_SIZE-1)/(BLOCK_SIZE)) N_work += i;
 cudaMalloc(&buff_d, N_work*sizeof(double)); // extra memory buffer for reduction across thread-blocks


 tgpu = omp_get_wtime();
 for (long counter=0; counter < N_REPEATS; counter++){
   cudaMemcpyAsync(a_d,a,M*N*sizeof(double),cudaMemcpyHostToDevice); //H2D
   cudaMemcpyAsync(b_d,b,N*sizeof(double),cudaMemcpyHostToDevice); //H2D
   cudaDeviceSynchronize();
   for (long row=0; row<M; row++){ //do row-wise vec vec on gpu
     double* sum_d = buff_d;
     Nb = (N+BLOCK_SIZE-1)/(BLOCK_SIZE);
     vecvec_kernel<<<Nb,BLOCK_SIZE>>>(sum_d, a_d+(row*N), b_d, N);
     while (Nb > 1) {
       long Nd = Nb;
       Nb = (Nb+BLOCK_SIZE-1)/(BLOCK_SIZE);
       reduction_kernel2<<<Nb,BLOCK_SIZE>>>(sum_d + Nd, sum_d, Nd);
       sum_d += Nd;
     }
     basicwrite<<<1,1>>>(sum_d,c_d+row);
   }
   cudaMemcpyAsync(c,c_d, M*sizeof(double), cudaMemcpyDeviceToHost);
   cudaDeviceSynchronize(); //synchronize blocks, i.e. wait for kernel
 }
 printf("mat-vec GPU Bandwidth = %f GB/s\n", 2*M*N*sizeof(double) / (omp_get_wtime()-tgpu)/(N_REPEATS*1e9));

 //mat-vec error
 err = 0;
 for (long i=0; i< M; i++) err += abs(c[i] - c_ref[i]);
 printf("mat-vec error: %1.2E\n",err);

 cudaFree(a_d);
 cudaFree(b_d);
 cudaFree(c_d);
 cudaFree(buff_d);

 free(a);
 free(b);
 free(c_ref);
 free(c);

  return 0;
}
