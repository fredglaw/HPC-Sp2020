#include <algorithm>
#include <stdio.h>
#include <math.h>
#include <omp.h>

#define N_THREADS 1

// Scan A array and write result into prefix_sum array;
// use long data type to avoid overflow
void scan_seq(long* prefix_sum, const long* A, long n) {
  if (n == 0) return;
  prefix_sum[0] = A[0]; // w[0] = A[0]
  for (long i = 1; i < n; i++) {
    prefix_sum[i] = prefix_sum[i-1] + A[i];
  }
}

void scan_omp(long* prefix_sum, long* A, long n) {
  long N = (n + N_THREADS - 1) / N_THREADS ; //set N=ceil(n/p)

  /* split up the full scan to the multiple threads
     note std::min(n,(i+1)*N) - i*N) should be N for all but the last thread
     if N is not divisible by N_THREADS */
  #pragma omp parallel for num_threads(N_THREADS)
  for (long i=0; i<N_THREADS; i++){
    scan_seq(prefix_sum+i*N, A+i*N, std::min(n,(i+1)*N) - i*N);
  }

  // update moving forward (serial for blocks, parallel within blocks)
  for (long j=0; j < N_THREADS -1; j++){
    long wj_end = prefix_sum[(j+1)*N-1]; //get term to update by
    #pragma omp parallel for num_threads(N_THREADS) //parallel for to update forward
    for (long k=(j+1)*N; k<(j+2)*N; k++){
      prefix_sum[k] += wj_end;
    }
  }
}

int main() {
  long N = 1e6; // problem size
  long N_repeats = 1000; // number of times to repeat experiment for timing
  double t_seq = 0; double tt_seq; //timers for sequential, singles and all
  double t_omp = 0; double tt_omp; //timers for omp, singles and all
  long err = 0; long temp_err; // errors, temp_err is error from each experiment
  for (long k=0; k<N_repeats; k++){
    //reallocate memory and reinitialize A completely random. Not included in timing
    long* A = (long*) malloc(N * sizeof(long));
    long* B0 = (long*) malloc(N * sizeof(long));
    long* B1 = (long*) malloc(N * sizeof(long));
    for (long i = 0; i < N; i++) A[i] = i; //initialize, different each k loop

    // scan sequentially for this one experiment
    tt_seq = omp_get_wtime();
    scan_seq(B0, A, N);
    t_seq += omp_get_wtime() - tt_seq;

    // scan in parallel for this one experiment
    tt_omp = omp_get_wtime();
    scan_omp(B1, A, N);
    t_omp += omp_get_wtime() - tt_omp;

    // compute error for this one experiment
    for (long i = 0; i < N; i++) temp_err = std::max(err, std::abs(B0[i] - B1[i]));
    err += temp_err;

    //free the A, B0, B1 used in this one experiment
    free(A);
    free(B0);
    free(B1);
  }

  double speedup = t_seq/t_omp; //compute the speedup
  double perc_theoretical_speedup = speedup/(double)N_THREADS; // % of optimal speedup

  printf("problem size: %e\n",(double)N);
  printf("# threads: %d\n",N_THREADS);
  printf("# experiment repeats for timing: %d\n",N_repeats);
  // printf("avg time for sequential-scan = %fs\n", t_seq/(double)N_repeats);
  // printf("avg time for parallel-scan = %fs\n", t_omp/(double)N_repeats);
  printf("tot time for sequential-scan = %fs\n", t_seq);
  printf("tot time for parallel-scan = %fs\n", t_omp);
  printf("error = %ld\n", err);
  printf("speedup: %f\n", speedup);
  printf("percentage of ideal speedup: %f\n",perc_theoretical_speedup);

  return 0;
}
