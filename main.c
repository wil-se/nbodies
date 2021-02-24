#include <stdio.h>
#include <stdlib.h> 
#include "lib-host.h"

int main(){
	printf("START =================================================\n");
	int n = set_memory();
	print_csv_bodies();
	bnode* root;
	root = (bnode*)malloc(sizeof(bnode));
	build_barnes_tree(root);
	
	print_tree(root);
	
	//compute_forces_all(root, 0);		
	print_csv_bodies();

	return 0;
}

