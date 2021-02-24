#include <stdio.h>
#include <stdlib.h>
#include "lib-host-common.h" 

int n;
double *x, *y, *z, *mass, *sx, *sy, *sz;

void print_csv_body(int i) {
        printf("%d,%lf,%lf,%lf,%lf,%lf,%lf,%lf\n", i, x[i], y[i], z[i], mass[i], sx[i], sy[i], sz[i]);
}

void print_csv_bodies(){
        for(int i=0; i<n; i++){
                print_csv_body(i);
        }
}

void set_memory(){
        scanf("%d", &n);
        x = (double*)malloc(sizeof(double)*n);
        y = (double*)malloc(sizeof(double)*n);
        z = (double*)malloc(sizeof(double)*n);
        mass = (double*)malloc(sizeof(double)*n);
        sx = (double*)malloc(sizeof(double)*n);
        sy = (double*)malloc(sizeof(double)*n);
        sz = (double*)malloc(sizeof(double)*n);
        for(int i=0; i<n; i++ ){
                scanf("%lf %lf %lf %lf %lf %lf %lf", &x[i], &y[i], &z[i], &mass[i], &sx[i], &sy[i], &sz[i]);
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


