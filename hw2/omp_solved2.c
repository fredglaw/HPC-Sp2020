/******************************************************************************
* FILE: omp_bug2.c
* DESCRIPTION:
*   Another OpenMP program with a bug.
* AUTHOR: Blaise Barney
* LAST REVISED: 04/06/05
******************************************************************************/
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main (int argc, char *argv[])
{
int nthreads, i;
double total=0.0;

/*** Spawn parallel region ***/
// make (i) private in the parallel section
#pragma omp parallel private(i)
  {
  /* Obtain thread number */
  int tid = omp_get_thread_num();
  /* Only master thread does this */
  if (tid == 0) {
    nthreads = omp_get_num_threads();
    printf("Number of threads = %d\n", nthreads);
    }
  printf("Thread %d is starting...\n",tid);

  #pragma omp barrier

  /* do some work */
  double my_total = 0.0; //make private variable for each thread to sum on
  #pragma omp for
  for (i=0; i<1000000; i++) //distribute work
     my_total = my_total + i*1.0;

  // race condition
  // critical section, only one thread increments public total at a time
  #pragma omp critical
  {
    printf("my turn in the critical, I am thread %d\n",tid);
    total += my_total;
  }

  #pragma omp barrier
  //wait till all threads are done incrementing to print
  printf ("Thread %d is done! Total= %e\n",tid,total);

  } /*** End of parallel region ***/
}
