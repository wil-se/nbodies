#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "common.h"
#include "sequential-barneshut.h"
#include "sequential-exhaustive.h"

int ITERS = 1000;

void exhaustive_openmp(){
    for(int i=0; i<ITERS; i++){
        print_csv_bodies();
        compute_ex_forces();
    }
}

void barneshut_openmp(){
    for(int i=0; i<ITERS; i++){
        print_csv_bodies();
        bnode* root;
	    root = (bnode*)malloc(sizeof(bnode));
	    build_barnes_tree(root);
        compute_barnes_forces_all(root, 1);
        destroy_barnes_tree(root);
    }
}