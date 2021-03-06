extern int n;
extern long  double *x, *y, *z, *mass, *sx, *sy, *sz;
extern long double *new_x, *new_y, *new_z, *new_sx, *new_sy, *new_sz;

void print_csv_body(int i);
void print_csv_bodies();
void set_memory();
void free_memory();
void set_new_memory();
void free_new_memory();
void set_new_vectors();
void set_vectors();
void print_plotlib(int body);
void print_plotlibs(int iter);
