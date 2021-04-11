#include <GL/glut.h>
#include "sequential-barneshut.h"

void display_seq_ex();
void display_seq_bh();
void processSpecialKeys(int key, int xx, int yy);
void mouseButton(int button, int state, int xcam, int ycam);
void mouseMove(int xcam, int ycam);
void reshape(GLint w, GLint h);
void timer(int v);
void render_sequential_exhaustive(int argc, char** argv);
void render_sequential_barneshut(int argc, char** argv);
void draw_axis();
void draw_body();
void display_tree(bnode* node);
void render_cuda_exhaustive(int argc, char** argv);
void display_cuda_ex();
