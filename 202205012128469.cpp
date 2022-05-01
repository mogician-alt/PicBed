#include <stdio.h>
#include <mpi.h>

/*   Open a file containing a vector, read its contents,
     and replicate the vector among all processes in a
     communicator. */

void read_replicated_vector (
   char        *s,      /* IN - File name */
   void       **v,      /* OUT - Vector */
   MPI_Datatype dtype,  /* IN - Vector type */
   int         *n,      /* OUT - Vector length */
   MPI_Comm     comm)   /* IN - Communicator */
{
   int        datum_size; /* Bytes per vector element */
   int        i;
   int        id;         /* Process rank */
   FILE      *infileptr;  /* Input file pointer */
   int        p;          /* Number of processes */

   MPI_Comm_rank (comm, &id);
   MPI_Comm_size (comm, &p);
   datum_size = get_size (dtype);
   if (id == (p-1)) {
      infileptr = fopen (s, "r");
      if (infileptr == NULL) *n = 0;
      else fread (n, sizeof(int), 1, infileptr);
   }
   MPI_Bcast (n, 1, MPI_INT, p-1, MPI_COMM_WORLD);
   if (! *n){
        if (!id) {
            printf ("Error: %s\n", "Cannot open vector file");
            fflush (stdout);
        }
        MPI_Finalize();
        exit (-1);
   }

   *v = my_malloc (id, *n * datum_size);

   if (id == (p-1)) {
      fread (*v, datum_size, *n, infileptr);
      fclose (infileptr);
   }
   MPI_Bcast (*v, *n, dtype, p-1, MPI_COMM_WORLD);
}


void read_row_striped_matrix (
   char        *s,        /* IN - File name */
   void      ***subs,     /* OUT - 2D submatrix indices */
   void       **storage,  /* OUT - Submatrix stored here */
   MPI_Datatype dtype,    /* IN - Matrix element type */
   int         *m,        /* OUT - Matrix rows */
   int         *n,        /* OUT - Matrix cols */
   MPI_Comm     comm)     /* IN - Communicator */
{
   int          datum_size;   /* Size of matrix element */
   int          i;
   int          id;           /* Process rank */
   FILE        *infileptr;    /* Input file pointer */
   int          local_rows;   /* Rows on this proc */
   void       **lptr;         /* Pointer into 'subs' */
   int          p;            /* Number of processes */
   void        *rptr;         /* Pointer into 'storage' */
   MPI_Status   status;       /* Result of receive */
   int          x;            /* Result of read */

   MPI_Comm_size (comm, &p);
   MPI_Comm_rank (comm, &id);
   datum_size = get_size (dtype);

   /* Process p-1 opens file, reads size of matrix,
      and broadcasts matrix dimensions to other procs */

   if (id == (p-1)) {
      infileptr = fopen (s, "r");
      if (infileptr == NULL) *m = 0;
      else {
         fread (m, sizeof(int), 1, infileptr);
         fread (n, sizeof(int), 1, infileptr);
      }      
   }
   MPI_Bcast (m, 1, MPI_INT, p-1, comm);

   if (!(*m)) MPI_Abort (MPI_COMM_WORLD, OPEN_FILE_ERROR);

   MPI_Bcast (n, 1, MPI_INT, p-1, comm);

   local_rows = BLOCK_SIZE(id,p,*m);

   /* Dynamically allocate matrix. Allow double subscripting
      through 'a'. */

   *storage = (void *) my_malloc (id,
       local_rows * *n * datum_size);
   *subs = (void **) my_malloc (id, local_rows * PTR_SIZE);

   lptr = (void *) &(*subs[0]);
   rptr = (void *) *storage;
   for (i = 0; i < local_rows; i++) {
      *(lptr++)= (void *) rptr;
      rptr += *n * datum_size;
   }

   /* Process p-1 reads blocks of rows from file and
      sends each block to the correct destination process.
      The last block it keeps. */

   if (id == (p-1)) {
      for (i = 0; i < p-1; i++) {
         x = fread (*storage, datum_size,
            BLOCK_SIZE(i,p,*m) * *n, infileptr);
         MPI_Send (*storage, BLOCK_SIZE(i,p,*m) * *n, dtype,
            i, DATA_MSG, comm);
      }
      x = fread (*storage, datum_size, local_rows * *n,
         infileptr);
      fclose (infileptr);
   } else
      MPI_Recv (*storage, local_rows * *n, dtype, p-1,
         DATA_MSG, comm, &status);
}


int main (int argc, char *argv[]) {
   double **a;       /* First factor, a matrix */
   double *b;        /* Second factor, a vector */
   double *c_block;  /* Partial product vector */
   double *c;        /* Replicated product vector */
   double    max_seconds;
   double    seconds;    /* Elapsed time for matrix-vector multiply */
   double *storage;  /* Matrix elements stored here */
   int    i, j;     /* Loop indices */
   int    id;       /* Process ID number */
   int    m;        /* Rows in matrix */
   int    n;        /* Columns in matrix */
   int    nprime;   /* Elements in vector */
   int    p;        /* Number of processes */
   int    rows;     /* Number of rows on this process */
   int    its;

   MPI_Init (&argc, &argv);
   MPI_Comm_rank (MPI_COMM_WORLD, &id);
   MPI_Comm_size (MPI_COMM_WORLD, &p);

   read_row_striped_matrix (argv[1], (void *) &a,
      (void *) &storage, MPI_DOUBLE, &m, &n, MPI_COMM_WORLD);
   rows = BLOCK_SIZE(id,p,m);
  //  print_row_striped_matrix ((void **) a, MPI_DOUBLE, m, n, MPI_COMM_WORLD);

   read_replicated_vector (argv[2], (void *) &b, MPI_DOUBLE,
      &nprime, MPI_COMM_WORLD);
   if (!id) { 
      for (int i = 0; i < nprime; i++) {
        printf ("%6.3f ", b[i]);
      }
      printf ("\n\n");
   }

   c_block = (double *) malloc (rows * sizeof(double));
   c = (double *) malloc (n * sizeof(double));
   MPI_Barrier (MPI_COMM_WORLD);
   seconds = - MPI_Wtime();
   for (i = 0; i < rows; i++) {
      c_block[i] = 0.0;
      for (j = 0; j < n; j++)
         c_block[i] += a[i][j] * b[j];
   }

   //transform a vector from a block distribution to a replicated distribution within a communicator

   int *cnt;  /* Elements contributed by each process */
   int *disp; /* Displacement in concatenated array */
   create_mixed_xfer_arrays (id, p, n, &cnt, &disp);
   MPI_Allgatherv (c_block, cnt[id], MPI_DOUBLE, c, cnt, disp, MPI_DOUBLE, MPI_COMM_WORLD);
   free (cnt);
   free (disp);

   MPI_Barrier (MPI_COMM_WORLD);
   seconds += MPI_Wtime();

   if (!id) { 
      for (int i = 0; i < n; i++) {
        printf ("%6.3f ", c[i]);
      }
      printf ("\n\n");
   }

   MPI_Allreduce (&seconds, &max_seconds, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
   if (!id) {
      printf ("MV1) N = %d, Processes = %d, Time = %12.6f sec,",
         n, p, max_seconds);
      printf ("Mflop = %6.2f\n", 2*n*n/(1000000.0*max_seconds));
   }
   MPI_Finalize();
   return 0;
}