#include <GL/glut.h>

void display();
void processSpecialKeys(int key, int xx, int yy);
void mouseButton(int button, int state, int xcam, int ycam);
void mouseMove(int xcam, int ycam);
void reshape(GLint w, GLint h);
void timer(int v);
void init_opengl(int argc, char** argv);
void draw_axis();
void draw_body();