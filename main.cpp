#ifdef __APPLE_CC__
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif
#include <cmath>
#include <stdio.h>

#define G 6.67e-11
#define dt 10

int n;
long  double *x, *y, *z, *new_x, *new_y, *new_z, *new_sx, *new_sy, *new_sz, *mass, *sx, *sy, *sz;



// angle of rotation for the camera direction
float angle=0.0f;
// actual vector representing the camera's direction
float lxcam=-0.0f,lzcam=-1.0f, lycam=0.0f;
// XZ position of the camera
float xcam=0.0f,zcam=40.0f,ycam=0.0f;
float deltaAngle = 0.0f;
int xOrigin = -1;
int yOrigin = -1;



void print_csv_body(int i) {
        printf("%d,%Lf,%Lf,%Lf,%Lf,%Lf,%Lf,%Lf\n", i, x[i], y[i], z[i], mass[i], sx[i], sy[i], sz[i]);
}

void print_csv_bodies(){
        for(int i=0; i<n; i++){
                print_csv_body(i);
        }
}

void set_memory(){
        scanf("%d", &n);
        x = (long double*)malloc(sizeof(long double)*n);
        y = (long double*)malloc(sizeof(long double)*n);
        z = (long double*)malloc(sizeof(long double)*n);
        mass = (long double*)malloc(sizeof(long double)*n);
        sx = (long double*)malloc(sizeof(long double)*n);
        sy = (long double*)malloc(sizeof(long double)*n);
        sz = (long double*)malloc(sizeof(long double)*n);
        for(int i=0; i<n; i++ ){
                scanf("%Lf %Lf %Lf %Lf %Lf %Lf %Lf", &x[i], &y[i], &z[i], &mass[i], &sx[i], &sy[i], &sz[i]);
        }
}

void free_memory(){
        free(x);
        free(y);
        free(z);
        free(mass);
        free(sx);
        free(sy);
        free(sz);
}

void set_new_memory(){
	new_x = (long double*)malloc(sizeof(long double)*n);
	new_y = (long double*)malloc(sizeof(long double)*n);
	new_z = (long double*)malloc(sizeof(long double)*n);
	new_sx = (long double*)malloc(sizeof(long double)*n);
	new_sy = (long double*)malloc(sizeof(long double)*n);
	new_sz = (long double*)malloc(sizeof(long double)*n);
}

void free_new_memory(){
	free(new_x);
	free(new_y);
	free(new_z);
	free(new_sx);
	free(new_sy);
	free(new_sz);
}

void set_new_vectors(){
	for(int i=0; i<n; i++){
        	new_x[i] = x[i];
        	new_y[i] = y[i];
        	new_z[i] = z[i];
        	new_sx[i] = sx[i];
        	new_sy[i] = sy[i];
        	new_sz[i] = sz[i];
	}       
}

void set_vectors(){
	for(int i=0; i<n; i++){
        	x[i] = new_x[i];
        	y[i] = new_y[i];
        	z[i] = new_z[i];
        	sx[i] = new_sx[i];
        	sy[i] = new_sy[i];
        	sy[i] = new_sz[i];
	}       
}


// forza applicata al corpo 2 esercitata dal corpo 1
void compute_ex_force(int body2, int body1){
	long double acc[3] = {0, 0, 0};
	long double force[3] = {0, 0, 0};
	long double distance[3] = {x[body2] - x[body1], y[body2] - y[body1], z[body2] - z[body1]};
	long double unit_vector[3] = {distance[0]/fabs(distance[0]), distance[1]/fabs(distance[1]), distance[2]/fabs(distance[2])};	

	force[0] = -G*((mass[body1]*mass[body2]/pow(distance[0], 2)))*unit_vector[0];
	force[1] = -G*((mass[body1]*mass[body2]/pow(distance[1], 2)))*unit_vector[1];
	force[2] = -G*((mass[body1]*mass[body2]/pow(distance[2], 2)))*unit_vector[2];
	
	acc[0] = force[0]/mass[body2];
	acc[1] = force[1]/mass[body2];
	acc[2] = force[2]/mass[body2];
	
	new_x[body1] += sx[body2]*dt + (acc[0])*dt*dt*0.5;
	new_y[body1] += sy[body2]*dt + (acc[1])*dt*dt*0.5;
	new_z[body1] += sz[body2]*dt + (acc[2])*dt*dt*0.5;
		
	long double new_acc[3] = {0, 0, 0};
	long double new_force[3] = {0, 0, 0};
	long double new_distance[3] = {new_x[body2] - x[body1], new_y[body2] - y[body1], new_z[body2] - z[body1]};
	long double new_unit_vector[3] = {new_distance[0]/fabs(new_distance[0]), new_distance[1]/fabs(new_distance[1]), new_distance[2]/fabs(new_distance[2])};
	
	new_force[0] = -G*((mass[body1]*mass[body2]/pow(new_distance[0], 2)))*new_unit_vector[0];
	new_force[1] = -G*((mass[body1]*mass[body2]/pow(new_distance[1], 2)))*new_unit_vector[1];
	new_force[2] = -G*((mass[body1]*mass[body2]/pow(new_distance[2], 2)))*new_unit_vector[2];

	new_acc[0] = new_force[0]/mass[body2];
	new_acc[1] = new_force[1]/mass[body2];
	new_acc[2] = new_force[2]/mass[body2];
	
	new_sx[body2] += 0.5*(acc[0] + new_acc[0])*dt;
	new_sy[body2] += 0.5*(acc[1] + new_acc[1])*dt;
	new_sz[body2] += 0.5*(acc[2] + new_acc[2])*dt;
}


void compute_ex_forces(){
	set_new_memory();	
	set_new_vectors();
	for(int i=0; i<n; i++){
		new_x[i] = x[i];
		new_y[i] = y[i];
		new_z[i] = z[i];
		new_sx[i] = sx[i];
		new_sy[i] = sy[i];
		new_sz[i] = sz[i];
	}	
	
	for(int j=0; j<n; j++){
		for(int k=0; k<n; k++){
			if(j != k){
				compute_ex_force(j, k);
			}
		}
	}
	set_vectors();
	free_new_memory();
}



// As usual, the routine to display the current state of the system is
// bracketed with a clearing of the window and a glFlush call.  Immediately
// within those calls the drawing code itself is bracketed by pushing and
// popping the current transformation.  And also as usual, we are assuming the
// current matrix mode is GL_MODELVIEW.  We finish with a SwapBuffers call
// because we'll animate.
void display() {
		compute_ex_forces();
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
    //glColor3f(5.0, 13.0, 43.0);
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

// The usual main() for a GLUT application.
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
