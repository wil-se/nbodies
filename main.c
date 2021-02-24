#include <stdio.h>
#include <stdlib.h> 
#include "lib-host.h"

int iters = 100;

int main(){
	printf("START =================================================\n");
	set_memory();
	print_csv_bodies();
	printf("\n");	
	for(int i=0; i<iters; i++){
		bnode* root;
		root = (bnode*)malloc(sizeof(bnode));
		build_barnes_tree(root);
		compute_forces_all(root, 1);
		destroy_barnes_tree(root);
		print_csv_bodies();
		printf("\n");
	}
	free_memory();
	
	return 0;
}

