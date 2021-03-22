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
  glutInit(&argc, argv);
  glutInitDisplayMode (GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
  glutInitWindowSize(1000, 1000);
  glutCreateWindow("space");
  glutDisplayFunc(display);
  glutReshapeFunc(reshape);
 //glutKeyboardFunc(processNormalKeys);
	glutSpecialFunc(processSpecialKeys);
  // here are the two new functions
	glutMouseFunc(mouseButton);
	glutMotionFunc(mouseMove);

  glutTimerFunc(100, timer, 0);
  glEnable(GL_DEPTH_TEST);
  glutMainLoop();

  free_memory();
}
