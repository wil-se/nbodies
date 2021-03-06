#include <stdio.h>
#include <stdlib.h> 
#include "lib-host-common.h"
#include "lib-host-barnes.h"
#include "lib-host-ex.h"


int iters = 10;

int main(){
	printf("START =================================================\n");
	set_memory();
	print_csv_bodies();
	
	//print_plotlibs(0);
	printf("\nBARNES: \n");
	for(int i=0; i<iters; i++){
		bnode* root;
		root = (bnode*)malloc(sizeof(bnode));
		build_barnes_tree(root);
		compute_barnes_forces_all(root, 1);
		destroy_barnes_tree(root);
		if(i == iters-1){print_csv_bodies();
		printf("\n");}
	}

	//print_csv_bodies();
	printf("EXHAUSTIVE:\n");
	for(int i=0; i<iters; i++){
		compute_ex_forces();
		if(i == iters-1){print_csv_bodies();
		printf("\n");}					
		//print_csv_bodies();
		//print_plotlibs(i);
	}
			
	free_memory();
	return 0;
}

