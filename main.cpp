// This program does a little solar system simulation from the point of view
// of a comet.  Here there is a yellow sun and a blue  planet; the planet
// spins on its own axis and it rotates around the sun.  The viewer is
// sitting on a comet and is always facing the sun.

#ifdef __APPLE_CC__
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif
#include <cmath>

// angle of rotation for the camera direction
float angle=0.0f;
// actual vector representing the camera's direction
float lx=-0.0f,lz=-1.0f, ly=0.0f;
// XZ position of the camera
float x=0.0f,z=5.0f,y=0.0f;
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
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glPushMatrix();

  // Draw sun: a yellow sphere of radius 1 centered at the origin.
  glColor3f(1.0, 1.0, 0.0);
  // myWireSphere(2.0, 16, 16);
  glutWireSphere(2.0, 16, 16);

  glTranslatef (1.0, 1.0, 1.0);
  glColor3f(5.0, 13.0, 43.0);
  // myWireSphere(1.8, 16.0, 16.0);
  glutWireSphere(2.0, 16.0, 16.0);

  glPopMatrix();
  glFlush();
  glutSwapBuffers();
}


void timer(int v) {
  glLoadIdentity();
  gluLookAt(x,y, z, x+lx,y+ly,z+lz, 0.0f,1.0f,0.0f);
  glutPostRedisplay();
  glutTimerFunc(1000/60, timer, v);
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
  gluPerspective(60.0, (GLfloat)w/(GLfloat)h, 1.0, 40.0);
  glMatrixMode(GL_MODELVIEW);
}

void processSpecialKeys(int key, int xx, int yy) {

	float fraction = 0.1f;

	switch (key) {
		case GLUT_KEY_LEFT :
			angle -= 0.01f;
			lx = sin(angle);
			lz = -cos(angle);
			break;
		case GLUT_KEY_RIGHT :
			angle += 0.01f;
			lx = sin(angle);
			lz = -cos(angle);
			break;
		case GLUT_KEY_UP :
			x += lx * fraction;
			z += lz * fraction;
			break;
		case GLUT_KEY_DOWN :
			x -= lx * fraction;
			z -= lz * fraction;
			break;
	}
}

void mouseButton(int button, int state, int x, int y) {

  // only start motion if the left button is pressed
  if (button == GLUT_LEFT_BUTTON) {

    // when the button is released
    if (state == GLUT_UP) {
      angle -= deltaAngle;
      xOrigin = -1;
      yOrigin = -1;
    }
    else {// state = GLUT_DOWN
      xOrigin = x;
      yOrigin = y;
    }
  }
}

void mouseMove(int x, int y) {
  if (xOrigin >= 0) {
    deltaAngle = (x - xOrigin) * 0.005f;
    lx = sin(angle - deltaAngle);
    lz = -cos(angle - deltaAngle);
  }
  if (yOrigin >= 0) {
    deltaAngle = (y - yOrigin) * 0.005f;
    //lz = sin(angle - deltaAngle);
    ly = -sin(angle - deltaAngle);
  }
}

// The usual main() for a GLUT application.
int main(int argc, char** argv) {
  glutInit(&argc, argv);
  glutInitDisplayMode (GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
  glutInitWindowSize(1000, 1000);
  glutCreateWindow("On a Comet");
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
}
