#include <stdio.h>
#include <stdlib.h> 
#include <cuda.h>
#include "common.h"
#include "render.h"
#include "device-common.h"
#include "cuda-exhaustive.h"

unsigned int omp_n_threads;

int main(int argc, char** argv) {
  set_memory();
  set_memory_cuda();

  omp_n_threads = atoi(argv[1]); 
  
  // cudaDeviceSynchronize();
  render_sequential_barneshut(argc, argv);
  // render_sequential_exhaustive(argc, argv);
  //render_cuda_exhaustive(argc, argv);
  cudaDeviceSynchronize();
  
  free_memory();
  return 0;
}
