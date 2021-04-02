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
float lxcam=-60.0f,lzcam=-60.0f, lycam=-30.0f;
// XZ position of the camera
float xcam=250.0f,zcam=250.0f,ycam=250.0f;
float deltaAngle = 0.0f;
int xOrigin = -1;
int yOrigin = -1;
float camera_speed = 50.0f;

void draw_axis(){
        glBegin(GL_LINES);

        glColor3f (255.0, 0.0, 0.0);
        glVertex3f(0.0, 0.0, 0.0);
        glVertex3f(10000000000000000.0, 0.0, 0.0);

        glColor3f (0.0, 255.0, 0.0);
        glVertex3f(0.0, 0.0, 0.0);
        glVertex3f(0.0, 10000000000000000.0, 0.0);

        glColor3f (0.0, 0.0, 255.0);
        glVertex3f(0.0, 0.0, 0.0);
        glVertex3f(0.0, 0.0, 10000000000000000.0);

        glColor3f (255.0, 0.0, 0.0);
        glVertex3f(0.0, 0.0, 0.0);
        glVertex3f(-10000000000000000.0, 0.0, 0.0);

        glColor3f (0, 255.0, 0.0);
        glVertex3f(0.0, 0.0, 0.0);
        glVertex3f(0.0, -10000000000000000.0, 0.0);

        glColor3f (0.0, 0.0, 255.0);
        glVertex3f(0.0, 0.0, 0.0);
        glVertex3f(0.0, 0.0, -10000000000000000.0);

        glEnd();
}

void draw_body(int i){
        glPushMatrix();
        glColor3f(100.0, 100.0, 100.0);  
        glTranslatef (x[i], y[i], z[i]);
        glutWireSphere(100, 16.0, 16.0);
        glPopMatrix();
}

void display_tree(bnode* node){
        queue* q = create_queue(1024);
        enqueue(q, node);
        glBegin(GL_LINES);

        while(q->size != 0){
                bnode* curr = dequeue(q);

                glColor3f (0, 250, 250);
                glVertex3f(curr->min_x, curr->min_y, curr->min_z);
                glVertex3f(curr->max_x, curr->min_y, curr->min_z);

                glColor3f (0, 250, 250);
                glVertex3f(curr->min_x, curr->min_y, curr->max_z);
                glVertex3f(curr->max_x, curr->min_y, curr->max_z);

                glColor3f (0, 250, 250);
                glVertex3f(curr->min_x, curr->max_y, curr->min_z);
                glVertex3f(curr->max_x, curr->max_y, curr->min_z);

                glColor3f (0, 250, 250);
                glVertex3f(curr->min_x, curr->max_y, curr->max_z);
                glVertex3f(curr->max_x, curr->max_y, curr->max_z);

                glColor3f (250, 0, 0);
                glVertex3f(curr->min_x, curr->min_y, curr->min_z);
                glVertex3f(curr->min_x, curr->max_y, curr->min_z);

                glColor3f (250, 0, 0);
                glVertex3f(curr->min_x, curr->min_y, curr->max_z);
                glVertex3f(curr->min_x, curr->max_y, curr->max_z);

                glColor3f (250, 0, 0);
                glVertex3f(curr->max_x, curr->min_y, curr->min_z);
                glVertex3f(curr->max_x, curr->max_y, curr->min_z);

                glColor3f (250, 0, 0);
                glVertex3f(curr->max_x, curr->min_y, curr->max_z);
                glVertex3f(curr->max_x, curr->max_y, curr->max_z);

                glColor3f (0, 250, 0);
                glVertex3f(curr->min_x, curr->min_y, curr->min_z);
                glVertex3f(curr->min_x, curr->min_y, curr->max_z);

                glColor3f (0, 250, 0);
                glVertex3f(curr->min_x, curr->max_y, curr->min_z);
                glVertex3f(curr->min_x, curr->max_y, curr->max_z);

                glColor3f (0, 250, 0);
                glVertex3f(curr->max_x, curr->min_y, curr->min_z);
                glVertex3f(curr->max_x, curr->min_y, curr->max_z);

                glColor3f (0, 250, 0);
                glVertex3f(curr->max_x, curr->max_y, curr->min_z);
                glVertex3f(curr->max_x, curr->max_y, curr->max_z);

                if(curr->body == -2){
                        enqueue(q, curr->o0);
                        enqueue(q, curr->o1);
                        enqueue(q, curr->o2);
                        enqueue(q, curr->o3);
                        enqueue(q, curr->o4);
                        enqueue(q, curr->o5);
                        enqueue(q, curr->o6);
                        enqueue(q, curr->o7);
                }
        }
        glEnd();
        destruct_queue(q);
}

void display_seq_ex() {
        print_csv_bodies();
        compute_ex_forces();
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        draw_axis();
        for(int i=0; i<n; i++){
                draw_body(i);
        }
        glFlush();
        glutSwapBuffers();
}

void display_seq_bh() {
        //print_csv_bodies();
        bnode* root;
        root = (bnode*)malloc(sizeof(bnode));
        build_barnes_tree(root);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        display_tree(root);
        compute_barnes_forces_all(root, 0);
        destroy_barnes_tree(root);
        //draw_axis();
        for(int i=0; i<n; i++){
                draw_body(i);
        }
        glFlush();
        glutSwapBuffers();
}


void timer(int v) {
        glLoadIdentity();
        gluLookAt(xcam,ycam, zcam, xcam+lxcam,ycam+lycam,zcam+lzcam, 0.0f,1.0f,0.0f);
        glutPostRedisplay();
        glutTimerFunc(1, timer, v);
}

void reshape(GLint w, GLint h) {
        glViewport(0, 0, w, h);
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        gluPerspective(90.0, (GLfloat)w/(GLfloat)h, 1.0, 10000000000000000000000000000.0);
        glMatrixMode(GL_MODELVIEW);
}

void processSpecialKeys(int key, int xx, int yy) {
        printf("LXCAM: %f LYCAM: %f\n", lxcam, lycam);
        switch (key) {
                case GLUT_KEY_LEFT :
                        xcam += lxcam * camera_speed;
                        ycam += lycam * camera_speed;
                        break;
                case GLUT_KEY_RIGHT :
                        xcam -= lxcam * camera_speed;
                        ycam -= lycam * camera_speed;
                        break;
                case GLUT_KEY_UP :
                        xcam += lxcam * camera_speed;
                        zcam += lzcam * camera_speed;
                        break;
                case GLUT_KEY_DOWN :
                        xcam -= lxcam * camera_speed;
                        zcam -= lzcam * camera_speed;
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
                lxcam = sin(angle - deltaAngle) * camera_speed;
                lzcam = -cos(angle - deltaAngle) * camera_speed;
        }
        if (yOrigin >= 0) {
                deltaAngle = (ycam - yOrigin) * 0.005f;
                lycam = tan(angle - deltaAngle) * camera_speed;
        }
}

void render_sequential_exhaustive(int argc, char** argv) {
        glutInit(&argc, argv);
        glutInitDisplayMode (GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
        glutInitWindowSize(1600, 900);
        glutCreateWindow("space");
        glutDisplayFunc(display_seq_ex);
        glutReshapeFunc(reshape);
        glutSpecialFunc(processSpecialKeys);
        glutMouseFunc(mouseButton);
        glutMotionFunc(mouseMove);
        glutTimerFunc(100, timer, 0);
        glEnable(GL_DEPTH_TEST);
        glutMainLoop();
}

void render_sequential_barneshut(int argc, char** argv) {
        glutInit(&argc, argv);
        glutInitDisplayMode (GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
        glutInitWindowSize(1600, 900);
        glutCreateWindow("space");
        glutDisplayFunc(display_seq_bh);
        glutReshapeFunc(reshape);
        glutSpecialFunc(processSpecialKeys);
        glutMouseFunc(mouseButton);
        glutMotionFunc(mouseMove);
        glutTimerFunc(100, timer, 0);
        glEnable(GL_DEPTH_TEST);
        glutMainLoop();
}
