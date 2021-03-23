#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "common.h"

int n;
long  double *x, *y, *z, *new_x, *new_y, *new_z, *new_sx, *new_sy, *new_sz, *mass, *sx, *sy, *sz;

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
        	sz[i] = new_sz[i];
	}       
}