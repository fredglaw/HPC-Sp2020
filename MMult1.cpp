// g++ -fopenmp -O2 -march=native MMult1.cpp && ./a.out

#include <stdio.h>
#include <math.h>
#include <omp.h>
#include "utils.h"

#define BLOCK_SIZE 16
#define N_THREADS 4

// Note: matrices are stored in column major order; i.e. the array elements in
// the (m x n) matrix C are stored in the sequence: {C_00, C_10, ..., C_m0,
// C_01, C_11, ..., C_m1, C_02, ..., C_0n, C_1n, ..., C_mn}
// void MMult0(long m, long n, long k, double *a, double *b, double *c) {
//   for (long j = 0; j < n; j++) {
//     for (long p = 0; p < k; p++) {
//       for (long i = 0; i < m; i++) {
//         double A_ip = a[i+p*m];
//         double B_pj = b[p+j*k];
//         double C_ij = c[i+j*m];
//         C_ij = C_ij + A_ip * B_pj;
//         c[i+j*m] = C_ij;
//       }
//     }
//   }
// }
void MMult0(long m, long n, long k, double *a, double *b, double *c) {
for (long j = 0; j < n; j++) {
  for (long p = 0; p < k; p++) {
    double B_pj = b[p+j*k];
    for (long i = 0; i < m; i++) {
        double A_ip = a[i+p*m];
        double C_ij = c[i+j*m];
        C_ij = C_ij + A_ip * B_pj;
        c[i+j*m] = C_ij;
      }
    }
  }
}

void MMult0_par(long m, long n, long k, double *a, double *b, double *c) {
#pragma omp parallel shared(a,b,c) num_threads(N_THREADS)
{
//note there are a number of factors that prevent a race condition. Namely, since
// j is the outer loop index, the j terms are split amongst the threads. That means
// no thread accesses the same j. Since the only race condition can happen at
// c[i+j*m], this automatically avoids it.
#pragma omp for
for (long j = 0; j < n; j++) {
  for (long p = 0; p < k; p++) {
    double B_pj = b[p+j*k];
    for (long i = 0; i < m; i++) {
        double A_ip = a[i+p*m];
        double C_ij = c[i+j*m];
        C_ij = C_ij + A_ip * B_pj;
        c[i+j*m] = C_ij;
      }
    }
  }
}
}


void MMult1(long m, long n, long k, double *a, double *b, double *c) {
  int M = m/BLOCK_SIZE;
  int N = n/BLOCK_SIZE;
  int K = m/BLOCK_SIZE;
  double C_IJ[BLOCK_SIZE*BLOCK_SIZE];
  double A_IP[BLOCK_SIZE*BLOCK_SIZE];
  double B_PJ[BLOCK_SIZE*BLOCK_SIZE];

  for (long I = 0; I < M; I++) {
    for (long J = 0; J < N; J++) {
      // read block C(i,j), load into cache
      for (long c2 = 0; c2 < BLOCK_SIZE; c2++){
        for (long c1 = 0; c1 < BLOCK_SIZE; c1++){
          C_IJ[c1+c2*BLOCK_SIZE] = c[BLOCK_SIZE*(I+J*m) + (c1 + c2*m)];
        }
      }
      for (long P = 0; P < K; P++) {
        // read blocks A(i,p), load into cache
        for (long a2 = 0; a2 < BLOCK_SIZE; a2++){
          for (long a1 = 0; a1 < BLOCK_SIZE; a1++){
            A_IP[a1+a2*BLOCK_SIZE] = a[BLOCK_SIZE*(I+P*m) + (a1 + a2*m)];
          }
        }
        // read block B(p,j), load into cache
        for (long b2 = 0; b2 < BLOCK_SIZE; b2++){
          for (long b1 = 0; b1 < BLOCK_SIZE; b1++){
            B_PJ[b1+b2*BLOCK_SIZE] = b[BLOCK_SIZE*(P+J*k) + (b1 + b2*k)];
          }
        }
        // Do block mat-mat
        for (long j2=0; j2 < BLOCK_SIZE; j2++){
          for (long p2=0; p2 < BLOCK_SIZE; p2++){
            double B_temp = B_PJ[p2+j2*BLOCK_SIZE];
            for (long i2=0; i2 < BLOCK_SIZE; i2++){
              C_IJ[i2+j2*BLOCK_SIZE] = C_IJ[i2+j2*BLOCK_SIZE] + A_IP[i2+p2*BLOCK_SIZE]*B_temp;
            }
          }
        }
      }
      // write block C(I,J)
      for (long c4 = 0; c4 < BLOCK_SIZE; c4++){
        for (long c3 = 0; c3 < BLOCK_SIZE; c3++){
          c[BLOCK_SIZE*(I+J*m) + (c3 + c4*m)] = C_IJ[c3+c4*BLOCK_SIZE];
        }
      }
    }
  }
}

int main(int argc, char** argv) {
  // const long PFIRST = BLOCK_SIZE;
  const long PFIRST = 128;
  // const long PFIRST = 600;
  const long PLAST = 1050;
  // const long PINC = std::max(50/BLOCK_SIZE,1) * BLOCK_SIZE; // multiple of BLOCK_SIZE
  const long PINC = 128;
  long NREPEATS = 10;

  printf(" Dimension       Time    Gflop/s       GB/s        Error\n");
  for (long p = PFIRST; p < PLAST; p += PINC) {
    long m = p, n = p, k = p;
    // long NREPEATS = 1e9 / (m*n*k) + 1;
    double* a = (double*) aligned_malloc(m * k * sizeof(double)); // m x k
    double* b = (double*) aligned_malloc(k * n * sizeof(double)); // k x n
    double* c = (double*) aligned_malloc(m * n * sizeof(double)); // m x n
    double* c_ref = (double*) aligned_malloc(m * n * sizeof(double)); // m x n

    // Initialize matrices
    for (long i = 0; i < m*k; i++) a[i] = drand48();
    for (long i = 0; i < k*n; i++) b[i] = drand48();
    for (long i = 0; i < m*n; i++) c_ref[i] = 0;
    for (long i = 0; i < m*n; i++) c[i] = 0;

    // NREPEATS = 1;
    for (long rep = 0; rep < NREPEATS; rep++) { // Compute reference solution
      MMult0(m, n, k, a, b, c_ref);
    }

    Timer t;
    t.tic();
    for (long rep = 0; rep < NREPEATS; rep++) {
      // MMult0(m,n,k, a, b, c); // comment out to time unlocked
      // MMult1(m,n,k, a, b, c); // comment out to time blocked
      MMult0_par(m,n,k, a, b, c); // comment out to time unlocked
    }
    double time = t.toc();
    //flops are the same block or unblocked, in serial or parallel
    double flops = NREPEATS*(2*m*n*k) / 1e9 / time; // TODO: calculate from m, n, k, NREPEATS, time

    //num mem accesses changes if blocked or unblocked
    // unblocked, (i,j,p) order
    // double bandwidth = NREPEATS*sizeof(double)*(2*m*n + 2*n*m*k) / 1e9 / time; // TODO: calculate from m, n, k, NREPEATS, time

    // unblocked, (j,p,i) order
    double bandwidth = NREPEATS*sizeof(double)*(n*k + 3*n*m*k) / 1e9 / time; // TODO: calculate from m, n, k, NREPEATS, time

    // blocked, optimal order
    // double bandwidth = NREPEATS*sizeof(double)*(2*m*n + 2*(m*n*k / BLOCK_SIZE)) / 1e9 / time; // TODO: calculate from m, n, k, NREPEATS, time
    printf("%10d %10f %10f %10f", p, time, flops, bandwidth);

    double max_err = 0;
    for (long i = 0; i < m*n; i++) max_err = std::max(max_err, fabs(c[i] - c_ref[i]));
    printf(" %10e\n", max_err);

    aligned_free(a);
    aligned_free(b);
    aligned_free(c);
  }


  return 0;
}

// * Using MMult0 as a reference, implement MMult1 and try to rearrange loops to
// maximize performance. Measure performance for different loop arrangements and
// try to reason why you get the best performance for a particular order?
//
//
// * You will notice that the performance degrades for larger matrix sizes that
// do not fit in the cache. To improve the performance for larger matrices,
// implement a one level blocking scheme by using BLOCK_SIZE macro as the block
// size. By partitioning big matrices into smaller blocks that fit in the cache
// and multiplying these blocks together at a time, we can reduce the number of
// accesses to main memory. This resolves the main memory bandwidth bottleneck
// for large matrices and improves performance.
//
// NOTE: You can assume that the matrix dimensions are multiples of BLOCK_SIZE.
//
//
// * Experiment with different values for BLOCK_SIZE (use multiples of 4) and
// measure performance.  What is the optimal value for BLOCK_SIZE?
//
//
// * Now parallelize your matrix-matrix multiplication code using OpenMP.
//
//
// * What percentage of the peak FLOP-rate do you achieve with your code?
//
//
// NOTE: Compile your code using the flag -march=native. This tells the compiler
// to generate the best output using the instruction set supported by your CPU
// architecture. Also, try using either of -O2 or -O3 optimization level flags.
// Be aware that -O2 can sometimes generate better output than using -O3 for
// programmer optimized code.
