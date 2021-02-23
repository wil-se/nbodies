#include <stdio.h>
#include <stdlib.h> 
#include "lib-host.h"

int main(){
	printf("START =================================================\n");
	int n = set_memory();
	
	bnode* root;
	root = (bnode*)malloc(sizeof(bnode));
	build_barnes_tree(root);
	
	print_tree(root);
		
	print_csv_bodies();

	return 0;
}

