// Parallel sample sort
// mpic++ -o ssort ssort.cpp && mpirun --use-hwthread-cpus -np 2 ./ssort
#include <stdio.h>
#include <unistd.h>
#include <mpi.h>
#include <stdlib.h>
#include <algorithm>

int main( int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  int rank, p;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &p);

  // Number of random numbers per processor (this should be increased
  // for actual tests or could be passed in through the command line
  int N = 1e4;

  int* vec = (int*)malloc(N*sizeof(int));
  // seed random number generator differently on every core
  srand((unsigned int) (rank + 393919));

  // fill vector with random integers
  for (int i = 0; i < N; ++i) {
    vec[i] = rand();
  }
  printf("rank: %d, first entry: %d\n", rank, vec[0]);

  double total_time = 0.0;
  MPI_Barrier(MPI_COMM_WORLD);
  total_time -= MPI_Wtime();
  // sort locally
  std::sort(vec, vec+N);


  // sample p-1 entries from vector as the local splitters, i.e.,
  // every N/P-th entry of the sorted vector

  int* s = (int*) malloc((p-1)*sizeof(int)); //local splitters
  // for (long k=0; k<p-1; k++) s[k] = vec[(k+1)*(N/p)-1]; //uniformly spaced
  for (long k=0; k<p-1; k++) s[k] = vec[(k+1)*(N/(p))]; //uniformly spaced


  // every process communicates the selected entries to the root
  // process; use for instance an MPI_Gather

  // root processor allocs space to receive
  int* root_gather = (int*) malloc(p*(p-1)*sizeof(int));

  //need to wait until root has alloc'd space before sending
  MPI_Barrier(MPI_COMM_WORLD);

  //gather to root
  MPI_Gather(s,p-1,MPI_INT,root_gather,p-1,MPI_INT,0,MPI_COMM_WORLD);

  // root process does a sort and picks (p-1) splitters (from the
  // p(p-1) received elements)
  if (0==rank){
    std::sort(root_gather, root_gather+(p*(p-1)));
    for (long k=0; k<p-1; k++){
      // s[k] = root_gather[(k+1)*(p-1)-1]; //uniformly spaced
      s[k] = root_gather[(k+1)*(p-1)]; //uniformly spaced
    }
  }





  // root process broadcasts splitters to all other processes
  MPI_Bcast(s,p-1,MPI_INT,0,MPI_COMM_WORLD); //overwrite s in all other processes

  MPI_Barrier(MPI_COMM_WORLD); //wait until Bcast is done
  // for(int k=0; k<p-1; k++) printf("procesor %d has local splitter s[%d] = %d\n",rank,k,s[k]);



  // every process uses the obtained splitters to decide which
  // integers need to be sent to which other process (local bins).
  // Note that the vector is already locally sorted and so are the
  // splitters; therefore, we can use std::lower_bound function to
  // determine the bins efficiently.
  //
  // Hint: the MPI_Alltoallv exchange in the next step requires
  // send-counts and send-displacements to each process. Determining the
  // bins for an already sorted array just means to determine these
  // counts and displacements. For a splitter s[i], the corresponding
  // send-displacement for the message to process (i+1) is then given by,
  // sdispls[i+1] = std::lower_bound(vec, vec+N, s[i]) - vec;

  int* num2send = (int*) malloc(p*sizeof(int)); //number to send to each process
  int* num2recv = (int*) malloc(p*sizeof(int)); //number to receive from each process
  int* sdispls = (int*) malloc(p*sizeof(int)); //displacements for sending
  int* rdispls = (int*) malloc(p*sizeof(int)); //displacements for receiving
  int totnum2recv = 0; //total number the process will receive

  // Do MPI_Alltoall to communicate how many integers each process should expect
  sdispls[0] = 0; //first displacement is 0
  for(long k=0; k<p-1; k++) sdispls[k+1] = std::lower_bound(vec, vec+N, s[k]) - vec;
  for(long k=0; k<p-1; k++) num2send[k] = sdispls[k+1] - sdispls[k]; //get num2send from displacements.
  num2send[p-1] = N - sdispls[p-1];
  MPI_Barrier(MPI_COMM_WORLD); //wait until above is done
  MPI_Alltoall(num2send,1,MPI_INT,num2recv,1,MPI_INT,MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD); //wait until process knows how many

  // Do MPI_Alltoallv to send the data
  for (long k=0; k<p; k++) totnum2recv += num2recv[k]; //get total number to receive
  // int* recv = (int*) malloc(totnum2recv*sizeof(int)); //each thread has its own to receive
  int* recv = (int*) calloc(totnum2recv, sizeof(int));
  // for (long k=0; k<p; k++) printf("processor %d will receive %d values from rank %ld\n",rank,num2recv[k],k);
  rdispls[0] = 0; //first displacement is 0
  for(long k=1; k<p; k++) rdispls[k] = rdispls[k-1]+num2recv[k-1]; //get displacements from num2recv
  // for(long k=0; k<p; k++) printf("processor %d will send at displacements: %d\n",rank,sdispls[k]);
  // for(long k=0; k<p; k++) printf("processor %d will receive at displacements: %d\n",rank,rdispls[k]);
  // printf("processor %d is receiving a total of %d values\n",rank,totnum2recv);

  MPI_Barrier(MPI_COMM_WORLD); //wait until above is done
  MPI_Alltoallv(vec,num2send,sdispls,MPI_INT,recv,num2recv,rdispls,MPI_INT,MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD); //wait until process knows how many

  // //debugging print statements
  // if (0==rank){
  //   for(long k=0; k<totnum2recv; k++) printf("processor 0 has received: %d\n",recv[k]);
  // }
  // MPI_Barrier(MPI_COMM_WORLD); //wait until process knows how many
  // for(long k=0; k<sdispls[1]; k++) printf("processor %d was suppose to send processor 0: %d\n",rank,vec[k]);
  // MPI_Barrier(MPI_COMM_WORLD); //wait until process knows how many
  // if (0==rank){
  //   for(long k=0; k<p; k++) printf("processor %d will receive at displacements: %d\n",rank,rdispls[k]);
  // }

  //
  // // send and receive: first use an MPI_Alltoall to share with every
  // // process how many integers it should expect, and then use
  // // MPI_Alltoallv to exchange the data
  //
  // // do a local sort of the received data
  std::sort(recv,recv+totnum2recv); //each processor now sorts its received lists
  MPI_Barrier(MPI_COMM_WORLD);
  total_time += MPI_Wtime();
  MPI_Barrier(MPI_COMM_WORLD);
  if (0==rank) printf("total time sorting %d values over %d processors was %f s\n",N,p,total_time);

  // every process writes its result to a file
  { // Write output to a file
    FILE* fd = NULL;
    char filename[256];
    snprintf(filename, 256, "output%02d.txt", rank);
    fd = fopen(filename,"w+");

    if(NULL == fd) {
      printf("Error opening file \n");
      return 1;
    }

    fprintf(fd, "rank %d has the sorted values:\n",rank);
    for(long k = 0; k < totnum2recv; k++)
      fprintf(fd, " %d\n", recv[k]);

    fclose(fd);
  }

  free(vec);
  free(s);
  // if(0==rank)
  free(root_gather);
  free(sdispls);
  free(num2send);
  free(num2recv);
  free(rdispls);
  // free(recv);
  MPI_Finalize();
  return 0;
}
