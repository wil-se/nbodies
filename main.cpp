#include <stdio.h>
#include <stdlib.h> 
#include "common.h"
#include "render.h"


int main(int argc, char** argv) {
  set_memory();
  render_sequential_barneshut(argc, argv);
  // render_sequential_exhaustive(argc, argv);
  free_memory();
}