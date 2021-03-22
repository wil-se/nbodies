#ifdef __APPLE_CC__
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

#include <stdio.h>
#include <stdlib.h> 
#include "common.h"
#include "render.h"


int main(int argc, char** argv) {
  set_memory();
  init_opengl(argc, argv);
  free_memory();
}
