/******************************************************************************
* FILE: omp_bug6.c
* DESCRIPTION:
*   This program compiles and runs fine, but produces the wrong result.
*   Compare to omp_orphan.c.
* AUTHOR: Blaise Barney  6/05
* LAST REVISED: 06/30/05
******************************************************************************/
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#define VECLEN 100

float a[VECLEN], b[VECLEN];

float dotprod (float *sum_pt)
{
int i,tid;
float my_sum = 0.0; //make a thread private copy to sum on

tid = omp_get_thread_num();
//distribute the work
#pragma omp for
  for (i=0; i < VECLEN; i++)
    {
    my_sum = my_sum + (a[i]*b[i]);
    printf("  tid= %d i=%d\n",tid,i);
    }
// race condition
// critical section, only one thread updates sum (via *sum_pt) at a time
#pragma omp critical
  *sum_pt = *sum_pt + my_sum;
}


int main (int argc, char *argv[]) {
int i;
float sum;
float *sum_pt = &sum; //make a pointer to access sum

for (i=0; i < VECLEN; i++)
  a[i] = b[i] = 1.0 * i;
sum = 0.0;

//make the pointer to sum now public
#pragma omp parallel shared(sum_pt)
  dotprod(sum_pt);

printf("Sum = %f\n",sum);

}
