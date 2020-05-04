/* MPI-parallel Jacobi smoothing to solve -u_xx - u_yy = f  with f=1
 * Global vector has N^2 unknowns, each processor works with its
 * part, which has lN^2 = N^2/p unknowns. So each processor deals with lN^2 terms
 * note, if there is a PMIX error, export environment variable
  export PMIX_MCA_gds=hash
 * This a modification on mpi12.cpp from lecture11, which does a 1D MPI Jacobi
 smoother with blocking. That code is by Georg Stadler
 * run using: mpic++ -o jacobi2D-mpi jacobi2D-mpi.cpp && mpirun --use-hwthread-cpus -np 4 ./jacobi2D-mpi 10 1000
 */
#include <stdio.h>
#include <math.h>
#include <mpi.h>
#include <string.h>

/* compuate global residual, assuming ghost values are updated */
double compute_residual(double *lu, int lN, double invhsq){
  int i,j;
  double tmp, gres = 0.0, lres = 0.0;

  //scan through non-ghost nodes and get residuals, assuming the ghost nodes are updated
  for (i = 1; i <= lN; i++){
    for (j = 1; j <= lN; j++){
      tmp = ((4.0*lu[i+j*(lN+2)] - lu[(i-1)+j*(lN+2)] - lu[(i+1)+j*(lN+2)] -
              lu[i+(j+1)*(lN+2)] - lu[i+(j-1)*(lN+2)]) * invhsq - 1);
      lres += tmp * tmp;
    }
  }
  /* use allreduce for convenience; a reduce would also be sufficient */
  MPI_Allreduce(&lres, &gres, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  return sqrt(gres);
}


int main(int argc, char * argv[]){
  int mpirank, i, j, p, N, lN, iter, max_iters;
  MPI_Status status, status1;

  if (argc < 3) {
    printf("Usage: mpirun ./jacobi2d-mpi (N) (max_iters)\n");
    abort();
  }

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
  MPI_Comm_size(MPI_COMM_WORLD, &p);
  int rootp = (int)sqrt(double(p)); //get square root of p

  /* get name of host running MPI process */
  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int name_len;
  MPI_Get_processor_name(processor_name, &name_len);
  printf("Rank %d/%d running on %s.\n", mpirank, p, processor_name);

  sscanf(argv[1], "%d", &N);
  sscanf(argv[2], "%d", &max_iters);

  /* compute number of unknowns handled by each process */
  lN = N / rootp;
  if ((N % rootp != 0) && mpirank == 0 ) {
    printf("N: %d, local N: %d\n", N, lN);
    printf("Exiting. N must be a multiple of p\n");
    MPI_Abort(MPI_COMM_WORLD, 0);
  }
  /* timing */
  MPI_Barrier(MPI_COMM_WORLD);
  double tt = MPI_Wtime();

  /* Allocation of vectors, including left/upper and right/lower ghost points */
  double * lu    = (double *) calloc(sizeof(double), (lN + 2)*(lN+2));
  double * lunew = (double *) calloc(sizeof(double), (lN + 2)*(lN+2));
  double * lutemp;

  /*use these buffers to load in the data to update ghost cells
   only write to the first lN, need +1 for the null terminating character */
  double * send_buff = (double*) calloc(sizeof(double), lN);
  double * recv_buff = (double*) calloc(sizeof(double), lN);

  double h = 1.0 / (N + 1);
  double hsq = h * h;
  double invhsq = 1./hsq;
  double gres, gres0, tol = 1e-5;

  MPI_Barrier(MPI_COMM_WORLD);
  /* initial residual */
  gres0 = compute_residual(lu, lN, invhsq);
  gres = gres0;
  if (0==mpirank){
    printf("gres0 = %f\n",gres);
  }
  MPI_Barrier(MPI_COMM_WORLD);

  double total_time = 0.0;
  MPI_Barrier(MPI_COMM_WORLD);
  total_time -= MPI_Wtime();

  for (iter = 0; iter < max_iters && gres/gres0 > tol; iter++) {

    /* Jacobi step for local points, using 4 point stencil*/
    for (i = 1; i <= lN; i++){
      for (j = 1; j <= lN; j++){
        lunew[i+j*(lN+2)]  = 0.25 * (hsq + lu[(i+1)+j*(lN+2)] + lu[(i-1)+j*(lN+2)] +
                                     lu[i+(j+1)*(lN+2)] + lu[i+(j-1)*(lN+2)]);
      }
    }


    /* communicate ghost values, use the fact that this is a grid of p = (rootp)^2
    processors. E.g. if p=9, rootp = 3:
                     _______ _______ _______
                    |       |       |       |
                    |   6   |   7   |   8   |
                    |_______|_______|_______|
                    |       |       |       |
                    |   3   |   4   |   5   |
                    |_______|_______|_______|
                    |       |       |       |
                    |   0   |   1   |   2   |
                    |_______|_______|_______|
    */
    if (mpirank%rootp < rootp - 1) {
      /* If not on the right side of the processor array, send right */
      for (long k=1; k<=lN; k++) send_buff[k-1] = lunew[lN+k*(lN+2)]; //load send buffer
      MPI_Send(send_buff, lN, MPI_DOUBLE, mpirank+1, 124, MPI_COMM_WORLD); //send right
      // printf("processor %d sending right to %d\n",mpirank,mpirank+1);
      MPI_Recv(recv_buff, lN, MPI_DOUBLE, mpirank+1, 123, MPI_COMM_WORLD, &status); //receive from right
      for (long k=1; k<=lN; k++) lunew[(lN+1)+k*(lN+2)] = recv_buff[k-1]; //write from recv buffer
      // printf("processor %d receiving right from %d\n",mpirank,mpirank+1);
    }
    if (mpirank%rootp > 0) {
      /* If not on the left side of the processor array, send left */
      for (long k=1; k<=lN; k++) send_buff[k-1] = lunew[1+k*(lN+2)]; //load send buffer
      MPI_Send(send_buff, lN, MPI_DOUBLE, mpirank-1, 123, MPI_COMM_WORLD); //send left
      // printf("processor %d sending left to %d\n",mpirank,mpirank-1);
      MPI_Recv(recv_buff, lN, MPI_DOUBLE, mpirank-1, 124, MPI_COMM_WORLD, &status); //receive from left
      for (long k=1; k<=lN; k++) lunew[(0)+k*(lN+2)] = recv_buff[k-1]; //write from recv buffer
      // printf("processor %d receiving left from %d\n",mpirank,mpirank-1);
    }
    if (mpirank/rootp < rootp - 1) {
      /* If not on the top of the processor array, send up */
      MPI_Send(&(lunew[1+lN*(lN+2)]), lN, MPI_DOUBLE, mpirank+rootp, 126, MPI_COMM_WORLD); //send up
      // printf("processor %d sending up to %d\n",mpirank,mpirank+rootp);
      MPI_Recv(&(lunew[1+(lN+1)*(lN+2)]), lN, MPI_DOUBLE, mpirank+rootp, 125, MPI_COMM_WORLD, &status); //receive from above
      // printf("processor %d receiving up from %d\n",mpirank,mpirank+rootp);
    }
    if (mpirank/rootp > 0) {
      /* If not on the bottom of the processor array, send down */
      MPI_Send(&(lunew[1+1*(lN+2)]), lN, MPI_DOUBLE, mpirank-rootp, 125, MPI_COMM_WORLD); //send down
      // printf("processor %d sending down to %d\n",mpirank,mpirank-rootp);
      MPI_Recv(&(lunew[1+0*(lN+2)]), lN, MPI_DOUBLE, mpirank-rootp, 126, MPI_COMM_WORLD, &status1); //receive from below
      // printf("processor %d receiving down from %d\n",mpirank,mpirank-rootp);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    /* copy newu to u using pointer flipping */
    // for (i=0; i<(lN+2)*(lN+2); i++) lu[i]=lunew[i];
    lutemp = lu; lu = lunew; lunew = lutemp;
    if (0 == (iter % 100)) { //change iter % k to print every k iterations
      gres = compute_residual(lu, lN, invhsq);
      if (0 == mpirank) {
	       printf("Iter %d: Residual: %6.2E\n", iter, gres);
      }
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);
  total_time += MPI_Wtime();
  MPI_Barrier(MPI_COMM_WORLD);
  if (0==mpirank){
    printf("solved using N_l = %d, with %d iterations\n",lN,max_iters);
    printf("total time using %d processors was %f s\n",p,total_time);
  }
  /* Clean up */
  free(lu);
  free(lunew);
  free(send_buff);
  free(recv_buff);

  /* timing */
  MPI_Barrier(MPI_COMM_WORLD);
  double elapsed = MPI_Wtime() - tt;
  if (0 == mpirank) {
    printf("Time elapsed is %f seconds.\n", elapsed);
  }
  MPI_Finalize();
  return 0;
}
