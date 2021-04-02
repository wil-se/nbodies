#include <stdio.h>
#include <stdlib.h> 
#include "common.h"
#include "render.h"
#include "cuda-barneshut.h"


int main(int argc, char** argv) {
  // set_memory();
  //render_sequential_barneshut(argc, argv);
  // render_sequential_exhaustive(argc, argv);
  simulate_bh_cuda();
  free_memory();
}
