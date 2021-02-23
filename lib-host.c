#include <stdio.h>
#include <stdlib.h>
#include "lib-host.h"
#include <math.h>

int n;
double *x, *y, *z, *mass, *sx, *sy, *sz;


queue* create_queue(int max_size){
        queue* q;
        q = (queue*)malloc(sizeof(queue));
        q->max_size = max_size;
        q->size = 0;
        q->head = NULL;
        q->tail = NULL;
        return q;
}

queue_node* create_queue_node(bnode* node){
        queue_node* n;
        n = (queue_node*)malloc(sizeof(queue_node));
        n->body = node;
        n->previous = NULL;
        return n;
}

void enqueue(queue* q, bnode* node){
        queue_node* q_node = create_queue_node(node);
        if(q->size >= q->max_size){
                return;
        }

        q_node->previous = NULL;

        if(q->size == 0){
                q->head = q_node;
                q->tail = q_node;
        }else{
                q->tail->previous = q_node;
                q->tail = q_node;
        }
        q->size++;

        return;
}

bnode* dequeue(queue* q){
        queue_node* head = q->head;
        bnode* node = head->body;
        q->head = head->previous;
        q->size--;
        free(head);
        return node;
}

void destruct_queue(queue* q){
        if(q->head == NULL){return;}
        queue_node* node = q->head;
        while(node->previous != NULL){
                queue_node* previous = node->previous;
                free(node);
                node = previous;
        }
        free(q);
}

void print_tree(bnode* node){
        queue* q = create_queue(1024);
        enqueue(q, node);
        while(q->size != 0){
                bnode* curr = dequeue(q);
                print_node(curr);
                if(curr->body == -2){
                        enqueue(q, curr->o0);
                        enqueue(q, curr->o1);
                        enqueue(q, curr->o2);
                        enqueue(q, curr->o3);
                        enqueue(q, curr->o4);
                        enqueue(q, curr->o5);
                        enqueue(q, curr->o6);
                        enqueue(q, curr->o7);
                }
        }
        destruct_queue(q);
}




void print_node(bnode* node){
        printf("================================\nBODY: %d\nDEPTH: %d\nMAX X: %lf\nMAX Y: %lf\nMAX Z: %lf\nMIN X: %lf\nMIN Y: %lf\nMIN Z: %lf\nX: %lf\nY: %lf\nZ: %lf\nMASS: %lf\n", node->body, node->depth, node->max_x, node->max_y, node->max_z, node->min_x, node->min_y, node->min_z, node->x, node->y, node->z, node->mass);
}

void print_csv_body(int i) {
        printf("%d,%lf,%lf,%lf,%lf,%lf,%lf,%lf\n", i, x[i], y[i], z[i], mass[i], sx[i], sy[i], sz[i]);
}

void print_csv_bodies(){
	for(int i=0; i<n; i++){
		print_csv_body(i);
	}
}

int set_memory(){
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
	return n;
}

double get_bound(){
	double max = 0;
        for(int i=0; i<n; i++){
		if(fabs(x[i]) > max){max = fabs(x[i]);}
                if(fabs(y[i]) > max){max = fabs(y[i]);}
                if(fabs(z[i]) > max){max = fabs(z[i]);}
        }
	return max;
}

void build_barnes_tree(bnode* root){
        double bound = get_bound();
	root->body = -1;
        root->depth = 0;
        root->max_x = bound;
        root->max_y = bound;
        root->max_z = bound;
        root->min_x = -bound;
        root->min_y = -bound;
        root->min_z = -bound;
        root->x = 0;
        root->y = 0;
        root->z = 0;
        root->mass = 0;
        generate_empty_children(root);

        for(int i=0; i<n; i++){
                //printf("first call body: %d x: %lf y: %lf z: %lf\n", i, x[i], y[i], z[i]);
		insert_body(root, i);
        }
}

void generate_empty_children(bnode *node){
        int depth = node->depth+1;
	double scalar = fabs(node->max_x - node->min_x)/2;
        bnode *o0, *o1, *o2, *o3, *o4, *o5, *o6, *o7;

        o0 = (bnode*)malloc(sizeof(bnode));
        o1 = (bnode*)malloc(sizeof(bnode));
        o2 = (bnode*)malloc(sizeof(bnode));
        o3 = (bnode*)malloc(sizeof(bnode));
        o4 = (bnode*)malloc(sizeof(bnode));
        o5 = (bnode*)malloc(sizeof(bnode));
        o6 = (bnode*)malloc(sizeof(bnode));
        o7 = (bnode*)malloc(sizeof(bnode));

        o0->depth = depth;
        o0->body = -1;
        o0->min_x = node->min_x + scalar;
        o0->max_x = node->max_x;
        o0->min_y = node->min_y + scalar;
        o0->max_y = node->max_y;
        o0->min_z = node->min_z + scalar;
        o0->max_z = node->max_z;

        o1->depth = depth;
        o1->body = -1;
        o1->min_x = node->min_x;
        o1->max_x = node->max_x - scalar;
        o1->min_y = node->min_y + scalar;
        o1->max_y = node->max_y;
        o1->min_z = node->min_z + scalar;
        o1->max_z = node->max_z;

        o2->depth = depth;
        o2->body = -1;
        o2->min_x = node->min_x;
        o2->max_x = node->max_x - scalar;
        o2->min_y = node->min_y;
        o2->max_y = node->max_y - scalar;
        o2->min_z = node->min_z + scalar;
        o2->max_z = node->max_z;

	o3->depth = depth;
	o3->body = -1;
	o3->min_x = node->min_x + scalar;
	o3->max_x = node->max_x;
	o3->min_y = node->min_y;
	o3->max_y = node->max_y - scalar;
	o3->min_z = node->min_z + scalar;
	o3->max_z = node->max_z;
	
	o4->depth = depth;
	o4->body = -1;
	o4->min_x = node->min_x + scalar;
	o4->max_x = node->max_x;
	o4->min_y = node->min_y + scalar;
	o4->max_y = node->max_y;
	o4->min_z = node->min_z;
	o4->max_z = node->max_z - scalar;
	
	o5->depth = depth;
	o5->body = -1;
	o5->min_x = node->min_x;
	o5->max_x = node->max_x - scalar;
	o5->min_y = node->min_y + scalar;
	o5->max_y = node->max_y;
	o5->min_z = node->min_z;
	o5->max_z = node->max_z - scalar;
	
	o6->depth = depth;
	o6->body = -1;
	o6->min_x = node->min_x;
	o6->max_x = node->max_x - scalar;
	o6->min_y = node->min_y;
	o6->max_y = node->max_y - scalar;
	o6->min_z = node->min_z;
	o6->max_z = node->max_z - scalar;
	
	o7->depth = depth;
	o7->body = -1;
	o7->min_x = node->min_x + scalar;
	o7->max_x = node->max_x;
	o7->min_y = node->min_y;
	o7->max_y = node->max_y - scalar;
	o7->min_z = node->min_z;
	o7->max_z = node->max_z - scalar;

        node->o0 = o0;
        node->o1 = o1;
        node->o2 = o2;
        node->o3 = o3;
        node->o4 = o4;
        node->o5 = o5;
        node->o6 = o6;
        node->o7 = o7;
}

bnode* get_octant(bnode* node, double x, double y, double z){

	double scalar = fabs(node->max_x - node->min_x)/2;

        bnode* result;
        //Q0
        if(node->min_x + scalar <= x && x <= node->max_x && node->min_y + scalar <= y && y <= node->max_y && node->min_z + scalar <= z && z <= node->max_z){
                result = node->o0;
        }
        //Q1
        if(node->min_x <= x && x <= node->max_x - scalar && node->min_y + scalar <= y && y <= node->max_y && node->min_z + scalar <= z && z <= node->max_z){
                result = node->o1;
        }

        //Q2
        if(node->min_x <= x && x <= node->max_x - scalar && node->min_y <= y && y <= node->max_y - scalar && node->min_z + scalar <= z && z <= node->max_z){
                result = node->o2;
        }

        //Q3
        if(node->min_x + scalar <= x && x <= node->max_x && node->min_y <= y && y <= node->max_y - scalar && node->min_z + scalar <= z && z <= node->max_z){
                result = node->o3;
        }

        //Q4
        if(node->min_x + scalar <= x && x <= node->max_x && node->min_y + scalar <= y && y <= node->max_y && node->min_z <= z && z <= node->max_z - scalar){
                result = node->o4;
        }

        //Q5
        if(node->min_x <= x && x <= node->max_x - scalar && node->min_y + scalar <= y && y <= node->max_y && node->min_z <= z && z <= node->max_z - scalar){
                result = node->o5;
        }

        //Q6
        if(node->min_x <= x && x <= node->max_x - scalar && node->min_y <= y && y <= node->max_y - scalar && node->min_z <= z && z <= node->max_z - scalar){
                result = node->o6;
        }

        //Q7
        if(node->min_x + scalar <= x && x <= node->max_x && node->min_y <= y && y <= node->max_y - scalar && node->min_z <= z && z <= node->max_z - scalar){
		result = node->o7;
        }
        return result;
}

void update(bnode* node, int body, double x, double y, double z, double mass){
	if(node->body >= 0){
                node->body = -2;
        }
        if(node->body == -1){
                node->body = body;
        }
        double tmass = node->mass + mass;
        double tx = ((node->mass*node->x)+(mass*x))/tmass;
	double ty = ((node->mass*node->y)+(mass*y))/tmass;
        double tz = ((node->mass*node->z)+(mass*z))/tmass;
        node->mass = tmass;
        node->x = tx;
        node->y = ty;
        node->z = tz;

}

void insert_body(bnode* node, int body){
	double bx = x[body], by = y[body], bz = z[body], bmass=mass[body];
	//printf("insert body: %d x: %lf y: %lf z: %lf\n", body, bx, by, bz);
	if(node->body == -1){
		update(node, body, bx, by, bz, bmass);
		return;
	}
	if(node->body >= 0){
		int old_body = node->body;
		generate_empty_children(node);
		bnode* next = get_octant(node, bx, by, bz);
		bnode* old_next = get_octant(node, x[old_body], y[old_body], z[old_body]);
		update(node, body, bx, by, bz, bmass);
		insert_body(old_next, old_body);
		insert_body(next, body);
		return;
	}
	if(node->body == -2){
        	update(node, body, bx, by, bz, bmass);
        	bnode* next = get_octant(node, bx, by, bz);
        	insert_body(next, body);
        	return;
	}

}


