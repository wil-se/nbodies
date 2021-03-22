#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "common.h"
#include "sequential-exhaustive.h"
#include "render.h"

#ifdef __APPLE_CC__
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

// angle of rotation for the camera direction
float angle=0.0f;
// actual vector representing the camera's direction
float lxcam=20.0f,lzcam=-90.0f, lycam=20.0f;
// XZ position of the camera
float xcam=0.0f,zcam=300.0f,ycam=0.0f;
float deltaAngle = 0.0f;
int xOrigin = -1;
int yOrigin = -1;


void display() {
		
    compute_ex_forces();
		print_csv_bodies();


  for(int i=0; i<n; i++){
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glPushMatrix();

	  if(i == 0){
		glColor3f(1.0, 1.0, 0.0);
	  }
	  if(i == 1){
		glColor3f(0.0, 0.0, 100.0);
	  }
	  
   glTranslatef (x[i], y[i], z[i]);
    glutWireSphere(2.0, 16.0, 16.0);
  glPopMatrix();
  glFlush();
  glutSwapBuffers();

  }

}


void timer(int v) {
  print_csv_bodies();
  glLoadIdentity();
  gluLookAt(xcam,ycam, zcam, xcam+lxcam,ycam+lycam,zcam+lzcam, 0.0f,1.0f,0.0f);
  glutPostRedisplay();
  glutTimerFunc(1, timer, v);
}

void reshape(GLint w, GLint h) {
  glViewport(0, 0, w, h);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluPerspective(90.0, (GLfloat)w/(GLfloat)h, 1.0, 10000.0);
  glMatrixMode(GL_MODELVIEW);
}

void processSpecialKeys(int key, int xx, int yy) {
  float fraction = 0.5f;

	switch (key) {
		case GLUT_KEY_LEFT :
			xcam += lxcam * fraction;
			ycam += lycam * fraction;
      break;
		case GLUT_KEY_RIGHT :
			xcam -= lxcam * fraction;
			ycam -= lycam * fraction;
      break;
		case GLUT_KEY_UP :
			xcam += lxcam * fraction;
			zcam += lzcam * fraction;
			break;
		case GLUT_KEY_DOWN :
			xcam -= lxcam * fraction;
			zcam -= lzcam * fraction;
			break;
	}
}

void mouseButton(int button, int state, int xcam, int ycam) {
  if (button == GLUT_LEFT_BUTTON) {
    if (state == GLUT_UP) {
      angle -= deltaAngle;
      xOrigin = -1;
      yOrigin = -1;
    }
    else {
      xOrigin = xcam;
      yOrigin = ycam;
    }
  }
}

void mouseMove(int xcam, int ycam) {
  if (xOrigin >= 0) {
    deltaAngle = (xcam - xOrigin) * 0.005f;
    lxcam = sin(angle - deltaAngle);
    lzcam = -cos(angle - deltaAngle);
  }
  if (yOrigin >= 0) {
    deltaAngle = (ycam - yOrigin) * 0.005f;
    lycam = -sin(angle - deltaAngle);
  }
}

void init_opengl(int argc, char** argv) {
  glutInit(&argc, argv);
  glutInitDisplayMode (GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
  glutInitWindowSize(1000, 1000);
  glutCreateWindow("space");
  glutDisplayFunc(display);
  glutReshapeFunc(reshape);
	glutSpecialFunc(processSpecialKeys);
	glutMouseFunc(mouseButton);
	glutMotionFunc(mouseMove);
  glutTimerFunc(100, timer, 0);
  glEnable(GL_DEPTH_TEST);
  glutMainLoop();
}