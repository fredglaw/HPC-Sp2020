// code by Frederick Law
// Script to solve Laplace equation in 2D using Gauss-Seidel iterations.
// Implement GS in black-red coloring
// g++ -std=c++11 -O3 gs2D-omp.cpp && ./a.out -N 100
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <omp.h>
#include "utils.h"
#define N_THREADS 4

//returns vector 2-norm of the residual
double norm_resid(double* u, double h_sq, long N, long Np2){
  double resid=0;
  #pragma omp parallel for reduction(+:resid) num_threads(N_THREADS) //collapse(2)
  for (long j=1; j <= N; j++){
    for (long i=1; i <= N; i++){
      resid += pow(((4*u[i+j*Np2]-u[(i-1)+j*Np2]-u[(i+1)+j*Np2]-
                        u[i+(j-1)*Np2]-u[i+(j+1)*Np2])/h_sq)-1 ,2.);
      }
  }
  return sqrt(resid);
}


double gs(double* u_curr, double* u_prev, double init_resid, double h_sq,
    long N, long Np2, long Nmax,double resid_terminate){
                long iter;
                double curr_resid;
                for (iter=0; iter<Nmax; iter++){

                  #pragma omp parallel num_threads(N_THREADS)
                  {
                  //keep the loop counters simple so parallelism is easier
                  // red sweeps
                  #pragma omp for //collapse(2)
                    for (long j=1; j <= N; j++){
                      for (long i=1; i <= N; i++){
                        if ((i+j)%2 == 0) //red labels have indices which always add to even
                          u_curr[i+j*Np2] = (h_sq + u_prev[(i-1)+j*Np2] + u_prev[(i+1)+j*Np2] +
                                                  u_prev[i+(j-1)*Np2] + u_prev[i+(j+1)*Np2])/4;
                      }
                    }
                  // deep copy the red sweeps to u_prev
                  #pragma omp for //collapse(2)
                    for (long j=1; j <= N; j++){
                      for (long i=1; i <= N; i++){
                        if((i+j)%2 == 0)
                          u_prev[i+j*Np2] = u_curr[i+j*Np2];
                      }
                    }

                  // black sweeps
                  #pragma omp for //collapse(2)
                    for (long j=1; j <= N; j++){
                      for (long i=1; i <= N; i++){
                        if ((i+j)%2 == 1) //red labels have indices which always add to odd
                        u_curr[i+j*Np2] = (h_sq + u_prev[(i-1)+j*Np2] + u_prev[(i+1)+j*Np2] +
                                                u_prev[i+(j-1)*Np2] + u_prev[i+(j+1)*Np2])/4;
                    }
                  }
                  // deep copy the black sweeps
                  #pragma omp for //collapse(2)
                    for (long j=1; j <= N; j++){
                      for (long i=1; i <= N; i++){
                        if((i+j)%2 == 1)
                          u_prev[i+j*Np2] = u_curr[i+j*Np2];
                      }
                    }
                  } //end parallel region for this update

                  curr_resid = norm_resid(u_curr,h_sq,N,Np2);

                  // Uncomment in order to print out residual after each loop (comment out for large Nmax)
                  // printf("Iteration: %d, Current Residual: %E\n",iter+1,h*curr_resid);


                  // if we hit residual termination condition, breathe loop
                  if ((curr_resid / init_resid) < resid_terminate){
                    break;
                  }
                }
                // printf("Final Iteration: %d, Current Residual: %E\n",iter,sqrt(h_sq)*curr_resid);
                return curr_resid;
}


int main(int argc,char *argv[])
{
  #ifdef _OPENMP
    printf("Using %d threads out of the maximum %d\n",N_THREADS, omp_get_max_threads());
  #endif
  long N; //declare N, either set as 100 by default or read in passed in argument
  if (argc < 3){
    N = 100;
    printf("No -N passed in, default is N=100\n");
  } else {
    N = read_option<long>("-N", argc, argv);
  }
  long Np2 = N+2;
  long Np2_sq = Np2*Np2;
  Timer t;

  long Nmax = 5e3;
  double h = 1/ ((double)N + 1);
  double h_sq = pow(h,2.);
  double resid_terminate = 1e-14; //termination condition on residual decrease
  // u is stored in COLUMN MAJOR ORDER, i.e. u_ij = u[i+j*Np2]
  double* u_curr = (double*)malloc(Np2*sizeof(double)*Np2*sizeof(double));
  double* u_prev = (double*)malloc(Np2*sizeof(double)*Np2*sizeof(double)); //secondary copy

  long N_REPEATS = 10;
  double time = 0; //count total time
  for (long num_times=0; num_times < N_REPEATS; num_times++){
    // initialize both as 0
    for (long i = 0; i < Np2_sq; i++){
      u_curr[i] = 0;
      u_prev[i] = 0;
    }
    double init_resid = norm_resid(u_curr,h,N,Np2); // get initial residual

    t.tic();
    gs(u_curr, u_prev, init_resid,h_sq,N,Np2,Nmax, resid_terminate);
    time += t.toc();
  }
  printf("%d Gauss-Seidel finished, avg time = %f\n",N_REPEATS,time/(double)N_REPEATS);
  // printf("Initial residual was: %f\n", h*init_resid);

  free(u_curr);
  free(u_prev);

  return 0;
}
