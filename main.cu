#include <stdio.h>
#include <stdlib.h> 
#include <cuda.h>
#include "common.h"
#include "render.h"
#include "device-common.h"
#include "cuda-exhaustive.h"
#include "cuda-barneshut.h"
#include "openmp.h"

int omp_num_th = 1;

int main(int argc, char** argv) {
        if (argc > 1) {
                int t = atoi(argv[1]);
                if (t > 0) omp_num_th = t;
        } 
        set_memory();
        // set_memory_cuda();

        // cudaDeviceSynchronize();
        // render_sequential_barneshut(argc, argv);
        // render_sequential_exhaustive(argc, argv);
        // render_cuda_exhaustive(argc, argv);
        // render_cuda_barneshut(argc, argv);
        // compute_barneshut_forces_cuda<<<1,1>>>();
        // cudaDeviceSynchronize();


        // OPENMP
        //exhaustive_openmp();
        barneshut_openmp();

        free_memory();
        return 0;
}
