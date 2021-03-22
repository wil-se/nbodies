extern int n;
extern long  double *x, *y, *z, *new_x, *new_y, *new_z, *new_sx, *new_sy, *new_sz, *mass, *sx, *sy, *sz;


void print_csv_body(int i);
void print_csv_bodies();
void set_memory();
void free_memory();
void set_new_memory();
void free_new_memory();
void set_new_vectors();
void set_vectors();
void compute_ex_force(int body2, int body1);
void compute_ex_forces();
