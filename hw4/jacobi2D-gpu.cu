// code by Frederick Law
// Script to solve Laplace equation in 2D using Jacobi iterations.
// nvcc -Xcompiler -fopenmp -O2 jacobi2D-gpu jacobi2D-gpu.cu && jacobi2D-gpu
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <omp.h>
#define N_THREADS 4 

void jacobi(double* u_curr, double* u_prev, double h_sq, long N, long Np2,
            long Nmax){
                for (long iter=0; iter<Nmax; iter++){
                  //create team of N_THREADS threads
                  #pragma omp parallel num_threads(N_THREADS)
                  {
                  //do single jacobi iteration
                  #pragma omp for //collapse(2)
                    for (long j=1; j <= N; j++){
                      for (long i=1; i <= N; i++){
                        u_curr[i+j*Np2] = (h_sq + u_prev[(i-1)+j*Np2] + u_prev[(i+1)+j*Np2] +
                                                  u_prev[i+(j-1)*Np2] + u_prev[i+(j+1)*Np2])/4;
                      }
                    }

                  //deep copy u_curr to u_prev
                  #pragma omp for //collapse(2)
                    for (long j=1; j <= N; j++){
                      for (long i=1; i <= N; i++){
                        u_prev[i+j*Np2] = u_curr[i+j*Np2];
                      }
                    }
                  } //end parallel region for this update
                } // end all iterations
}



//kernel which executes the jacobi update
__global__ void jacobi_iterate_kernel(double* u_curr, double* u_prev, double h_sq,
                              long N, long Np2, long Nmax){
          long idx = blockIdx.x * blockDim.x + threadIdx.x; //get thread id
          long idx_j = (idx/N) + 1; // floor of (idx/N) + 1 is i index
          long idx_i = (idx%N) + 1; // remainder of (idx/N) + 1 is j index
          //do single jacobi iteration
          if(idx<N*N) u_curr[idx_i+idx_j*Np2] = (h_sq + u_prev[(idx_i-1)+idx_j*Np2] + u_prev[(idx_i+1)+idx_j*Np2] +
                                      u_prev[idx_i+(idx_j-1)*Np2] + u_prev[idx_i+(idx_j+1)*Np2])/4;

}

//kernel to copy u_prev to u_curr
__global__ void jacobi_copy_kernel(double* u_curr, double* u_prev, double h_sq,
                              long N, long Np2, long Nmax){
          long idx = blockIdx.x * blockDim.x + threadIdx.x; //get thread id
          long idx_j = (idx/N) + 1; // floor of (idx/N) + 1 is i index
          long idx_i = (idx%N) + 1; // remainder of (idx/N) + 1 is j index
          //do single jacobi iteration
          if(idx<N*N) u_prev[idx_i+idx_j*Np2] = u_curr[idx_i+idx_j*Np2];
}




//CUDA error check
void Check_CUDA_Error(const char *message){
  cudaError_t error = cudaGetLastError();
  if(error!=cudaSuccess) {
    fprintf(stderr,"ERROR: %s: %s\n", message, cudaGetErrorString(error) );
    exit(-1);
  }
}


int main(int argc,char *argv[])
{
  long N=2048; //declare N, either set as 100 by default or read in passed in argument
  long N_blocks = ((N*N-1)/1024)+1;
  long N_threads = min((long)1024,N*N);
  long Np2 = N+2;
  long Np2_sq = Np2*Np2;

  long Nmax = 5e3;
  double h = 1/ ((double)N + 1);
  double h_sq = pow(h,2.);

  // u is stored in COLUMN MAJOR ORDER, i.e. u_ij = u[i+j*Np2]
  double* u_curr_ref = (double*)malloc(Np2*sizeof(double)*Np2*sizeof(double));
  double* u_prev_ref = (double*)malloc(Np2*sizeof(double)*Np2*sizeof(double)); //secondary copy
  double* u_curr = (double*)malloc(Np2*sizeof(double)*Np2*sizeof(double));
  double* u_prev = (double*)malloc(Np2*sizeof(double)*Np2*sizeof(double)); //secondary copy

  //cudaMalloc space
  double *u_curr_d, *u_prev_d; //declare pointers for memcopy to device
  cudaMalloc(&u_curr_d, Np2*sizeof(double)*Np2*sizeof(double));
  Check_CUDA_Error("malloc u_curr_d failed");
  cudaMalloc(&u_prev_d, Np2*sizeof(double)*Np2*sizeof(double));

  double error = 0;
  double tcpu;
  double tcpuTotal = 0;
  double tgpu;
  double tgpuTotal = 0;
  long N_REPEATS = 10;
  for (long num_times=0; num_times < N_REPEATS; num_times++){
    // initialize both as 0
    for (long i = 0; i < Np2_sq; i++){
      u_curr_ref[i] = 0;
      u_prev_ref[i] = 0;
      u_curr[i] = 0;
      u_prev[i] = 0;
    }

    tcpu = omp_get_wtime();
    jacobi(u_curr_ref, u_prev_ref,h_sq,N,Np2,Nmax);
    tcpuTotal += omp_get_wtime() - tcpu;

    tgpu = omp_get_wtime();
    cudaMemcpy(u_curr_d,u_curr,Np2*sizeof(double)*Np2*sizeof(double),cudaMemcpyHostToDevice); //H2D
    cudaMemcpy(u_prev_d,u_prev,Np2*sizeof(double)*Np2*sizeof(double),cudaMemcpyHostToDevice); //H2D
    for(long iter=0; iter< Nmax; iter++){
      jacobi_iterate_kernel<<<N_blocks,N_threads>>>(u_curr_d,u_prev_d,h_sq,N,Np2,Nmax);
      jacobi_copy_kernel<<<N_blocks,N_threads>>>(u_curr_d,u_prev_d,h_sq,N,Np2,Nmax);
    }
    cudaDeviceSynchronize();
    cudaMemcpy(u_curr, u_curr_d,Np2*sizeof(double)*Np2*sizeof(double),cudaMemcpyDeviceToHost); //D2H
    tgpuTotal += omp_get_wtime() - tgpu;

    //for (long j=0; j<Np2_sq; j++){
    //  error = max(abs(u_curr[j] - u_curr_ref[j]),error);
    //}
  }

  //compute error;
  //double error=0;
  for (long i=0; i<Np2_sq; i++){
    error = max(abs(u_curr[i] - u_curr_ref[i]), error);
    //printf("solution at %d is %f\n",i,u_curr[i]);
    //printf("reference at %d is %f\n",i,u_curr_ref[i]);
  }
  //printf("num blocks is: %d, num threads per block is: %d\n",N_blocks,N_threads);
  //printf("h^2 / 4 is: %f\n",h_sq/4);
  printf("%d Jacobi finished, CPU avg time = %f\n",N_REPEATS,tcpuTotal/(double)N_REPEATS);
  printf("%d Jacobi finished, GPU avg time = %f\n",N_REPEATS,tgpuTotal/(double)N_REPEATS);
  printf("error is: %E\n",error);

  //clean up
  cudaFree(u_curr_d);
  cudaFree(u_prev_d);

  free(u_curr_ref);
  free(u_prev_ref);
  free(u_curr);
  free(u_prev);

  return 0;
}
