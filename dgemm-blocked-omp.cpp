#include <iostream>
#include <string.h>
#include <stdlib.h>
#include <omp.h>
#include "likwid-stuff.h"
#include <algorithm>

const char *dgemm_desc = "Blocked dgemm, OpenMP-enabled";

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are n-by-n matrices stored in row-major format.
 * On exit, A and B maintain their input values. */
void square_dgemm_blocked(int n, int block_size, double *A, double *B, double *C)
{
   // insert your code here: implementation of blocked matrix multiply with copy optimization and OpenMP parallelism enabled

   // be sure to include LIKWID_MARKER_START(MY_MARKER_REGION_NAME) inside the block of parallel code,
   // but before your matrix multiply code, and then include LIKWID_MARKER_STOP(MY_MARKER_REGION_NAME)
   // after the matrix multiply code but before the end of the parallel code block.

   std::cout << "Insert your blocked matrix multiply with copy optimization, openmp-parallel edition here " << std::endl;

   int N = (n + block_size - 1) / block_size; // number of blocks along one dimension

#pragma omp parallel
   {
      // local copies of blocks for each thread
      double *matrixCopyA = new double[block_size * block_size];
      double *matrixCopyB = new double[block_size * block_size];
      double *matrixCopyC = new double[block_size * block_size];

#pragma omp barrier // wait for all threads to join
      LIKWID_MARKER_START(MY_MARKER_REGION_NAME);
#pragma omp for collapse(2) schedule(static)
      for (int i = 0; i < N; i++)
      {
         for (int j = 0; j < N; j++)
         {
            // create local copies of C block
            int startRowC = i * block_size;
            int startColC = j * block_size;

            // determine actual block sizes (may be smaller at edges，
            // this process idea come from ai-assisted suggestion)
            // which is a little different from previous version CP2
            int ib = std::min(block_size, n - startRowC);
            int jb = std::min(block_size, n - startColC);

            // copy C block into local matrixCopyC
            for (int relativeRow = 0; relativeRow < ib; relativeRow++)
            {
               memcpy(matrixCopyC + relativeRow * block_size,
                      C + (startRowC + relativeRow) * n + startColC,
                      jb * sizeof(double));
            }

            // Multiply blocks of A and B, accumulate into local matrixCopyC
            for (int k = 0; k < N; k++)
            {
               int startRowA = i * block_size;
               int startColA = k * block_size;

               int startRowB = k * block_size;
               int startColB = j * block_size;

               // determine actual block sizes
               int kb = std::min(block_size, n - startColA);

               // copy A block into local matrixCopyA
               for (int relativeRow = 0; relativeRow < ib; relativeRow++)
               {
                  memcpy(matrixCopyA + relativeRow * block_size,
                         A + (startRowA + relativeRow) * n + startColA,
                         kb * sizeof(double));
               }
               // copy B block into local matrixCopyB
               for (int relativeRow = 0; relativeRow < kb; relativeRow++)
               {
                  memcpy(matrixCopyB + relativeRow * block_size,
                         B + (startRowB + relativeRow) * n + startColB,
                         jb * sizeof(double));
               }

               // Perform the block multiplication and accumulate results
               for (int i = 0; i < ib; ++i)
               {
                  for (int j = 0; j < jb; ++j)
                  {
                     for (int k = 0; k < kb; ++k)
                     {
                        matrixCopyC[i * block_size + j] +=
                            matrixCopyA[i * block_size + k] *
                            matrixCopyB[k * block_size + j];
                     }
                  }
               }
            }
            // copy local matrixCopyC back into C block
            for (int relativeRow = 0; relativeRow < ib; relativeRow++)
            {
               memcpy(C + (startRowC + relativeRow) * n + startColC,
                      matrixCopyC + relativeRow * block_size,
                      jb * sizeof(double));
            }
         }
      }
#pragma omp barrier // end of parallel region
      LIKWID_MARKER_STOP(MY_MARKER_REGION_NAME);

      delete[] matrixCopyA;
      delete[] matrixCopyB;
      delete[] matrixCopyC;
   }
}
