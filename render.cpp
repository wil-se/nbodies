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
float lxcam=-0.0f,lzcam=-1.0f, lycam=0.0f;
// XZ position of the camera
float xcam=0.0f,zcam=40.0f,ycam=0.0f;
float deltaAngle = 0.0f;
int xOrigin = -1;
int yOrigin = -1;


// As usual, the routine to display the current state of the system is
// bracketed with a clearing of the window and a glFlush call.  Immediately
// within those calls the drawing code itself is bracketed by pushing and
// popping the current transformation.  And also as usual, we are assuming the
// current matrix mode is GL_MODELVIEW.  We finish with a SwapBuffers call
// because we'll animate.
void display() {
		
    //compute_ex_forces();
		print_csv_bodies();

  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glPushMatrix();

  for(int i=0; i<n; i++){
	  if(i == 0){
		glColor3f(1.0, 1.0, 0.0);
	  }
	  if(i == 1){
		glColor3f(0.0, 0.0, 100.0);
	  }
	  
   glTranslatef (x[i], y[i], z[i]);
    glutWireSphere(2.0, 16.0, 16.0);
  }

  //Draw sun: a yellow sphere of radius 1 centered at the origin.
  //glColor3f(1.0, 1.0, 0.0);
  //glutWireSphere(2.0, 16, 16);

  //glTranslatef (1.0, 1.0, 1.0);
  //glColor3f(5.0, 13.0, 43.0);
  //glutWireSphere(2.0, 16.0, 16.0);

  glPopMatrix();
  glFlush();
  glutSwapBuffers();
}


void timer(int v) {
  print_csv_bodies();
  glLoadIdentity();
  gluLookAt(xcam,ycam, zcam, xcam+lxcam,ycam+lycam,zcam+lzcam, 0.0f,1.0f,0.0f);
  glutPostRedisplay();
  glutTimerFunc(1, timer, v);
}

// As usual we reset the projection transformation whenever the window is
// reshaped.  This is done (of course) by setting the current matrix mode
// to GL_PROJECTION and then setting the matrix.  It is easiest to use the
// perspective-projection-making matrix from the GL utiltiy library.  Here
// we set a perspective camera with a 60-degree vertical field of view,
// an aspect ratio to perfectly map into the system window, a near clipping
// plane distance of 1.0 and a far clipping distance of 40.0.  The last
// thing done is to reset the current matrix mode to GL_MODELVIEW, as
// that is expected in all the calls to display().
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
			//angle -= 0.01f;
			//lx = sin(angle);
			//lz = -cos(angle);
			xcam += lxcam * fraction;
			ycam += lycam * fraction;
      break;
		case GLUT_KEY_RIGHT :
			//angle += 0.01f;
			//lx = sin(angle);
			//lz = -cos(angle);
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

  // only start motion if the left button is pressed
  if (button == GLUT_LEFT_BUTTON) {

    // when the button is released
    if (state == GLUT_UP) {
      angle -= deltaAngle;
      xOrigin = -1;
      yOrigin = -1;
    }
    else {// state = GLUT_DOWN
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
    //lz = sin(angle - deltaAngle);
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