#include <cuda.h>
#include <stdio.h>
#include "device-common.h"
#include "cuda-barneshut.h"
#define G 6.67e-11
#define dt 1000000


__device__ int *semaphore, *semaphore_mutex;

__global__ void compute_barneshut_forces_cuda(){
    int block_amount = (n_d/512)+1;
    int block_dim = n_d/block_amount;
    
    double *max_x, *max_y, *max_z;
    cudaMalloc((void**)&max_x, sizeof(double));
    cudaMalloc((void**)&max_y, sizeof(double));
    cudaMalloc((void**)&max_z, sizeof(double));

    get_max_x_cuda<<<block_amount, block_dim, n_d*sizeof(double)>>>(max_x);
    get_max_y_cuda<<<block_amount, block_dim, n_d*sizeof(double)>>>(max_y);
    get_max_z_cuda<<<block_amount, block_dim, n_d*sizeof(double)>>>(max_z);

    cudaDeviceSynchronize();
    
    double max = 0;
    if (*max_x > *max_y) {
        if (*max_x > *max_z){
            max = *max_x;
        } else {
            max = *max_z;
        }
    } else {
        if (*max_y > *max_z) {
            max = *max_y;
        } else {
            max = *max_z;
        }
    }

    printf("max: %lf\n", max);

    bnode_cuda* root;
	cudaMalloc((void**)&root, sizeof(bnode_cuda));

    root->body = -1;
    root->depth = 0;
    cudaMalloc((void**)&root->mutex, sizeof(int));
    *root->mutex = 0;
    root->max_x = max;
    root->max_y = max;
    root->max_z = max;
    root->min_x = -max;
    root->min_y = -max;
    root->min_z = -max;
    root->x = 0;
    root->y = 0;
    root->z = 0;
    root->mass = 0;
    // print_node_cuda(root);
    generate_empty_children_cuda(root);

    cudaMalloc((void**)&semaphore, sizeof(int));
    *semaphore = n_d;
    cudaMalloc((void**)&semaphore_mutex, sizeof(int));
    *semaphore_mutex = 0;
    // printf("\n\nsem value: %d\n\n", *semaphore);
    
    
    insert_body_cuda<<<block_amount, block_dim>>>(root);
    cudaDeviceSynchronize();
    print_node_cuda(root);

}


__global__ void insert_body_cuda(bnode_cuda *root){
    int body = (blockIdx.x * blockDim.x) + threadIdx.x;
    bnode_cuda *parent = root;
    double bx = x_d[body], by = y_d[body], bz = z_d[body], bmass = mass_d[body];
    bnode_cuda *node;
    // printf("hello from thread: %d\n", body);
    int locked = 1;
    while(locked){
        if(atomicCAS(root->mutex, 0, 1) == 0){
            update_cuda(parent, body, bx, by, bz, bmass);
            atomicExch(root->mutex, 0);
            locked = 0;
        }
    }
    
    node = get_octant_cuda(parent, bx, by, bz);
    __syncthreads();

    int wait = 0;
    while(*semaphore != 0){
        locked = 1;
        __syncthreads();
        if(wait == 1 && node->body >= 0){
            printf("body %d is waiting others to finish.. sem value: %d\n", body, *semaphore);
            continue;
        }
        if(wait == 1 && node->body == -2){
            printf("some body updated body %d node.. were waiting: %d.. node->body = %d\n", body, wait, node->body);
            increment_semaphore();
            wait = 0;
            parent = node;
            node = get_octant_cuda(parent, bx, by, bz);
        }
    
        while(locked){
            printf("body %d waiting to acquire lock, value: %d node->body value is %d\n", body, *(node->mutex), node->body);
            if(atomicCAS(node->mutex, 0, 1) == 0){

                if(node->body ==  -1){
                    printf("body %d found an empty leaf\n", body);
                    update_cuda(node, body, bx, by, bz, bmass);
                    decrement_semaphore();
                    atomicExch(node->mutex, 0);
                    wait = 1;
                    locked = 0;
                    break;
                }

                if(node->body >= 0){
                    printf("body %d found a full leaf\n", body);
                    update_cuda(node, body, bx, by, bz, bmass);
                    generate_empty_children_cuda(node);
                    atomicExch(node->mutex, 0);

                    parent = node;
                    node = get_octant_cuda(parent, bx, by, bz);
                    locked = 0;
                    break;
                }

                if(node->body == -2){
                    printf("body %d found an internal node\n", body);
                    update_cuda(node, body, bx, by, bz, bmass);
                    atomicExch(node->mutex, 0);

                    parent = node;
                    node = get_octant_cuda(parent, bx, by, bz);
                    locked = 0;
                    break;
                }
                
            }
        }
        printf("thread %d ended a cycle\n", body);
    }

    printf("bye\n");























    // while(*semaphore != 0){
    //     // printf("body %d getting octant..\n", body);
    //     // start critical section
    //     locked = 1;
    //     printf("body %d out of critical section..\n", body);
    //     while(locked && *semaphore != 0 ){
    //         printf("body %d waiting to acquire lock.. %d\n", body, *(node->mutex));
    //         if(atomicCAS(node->mutex, 0, 1) == 0) {
    //             printf("body %d passed lock acquire..\n", body);
    //             if(node->body == -2){
    //                 printf("body %d found an internal node\n", body);
    //                 update_cuda(node, body, bx, by, bz, bmass);
    //                 parent = node;
    //                 node = get_octant_cuda(parent, bx, by, bz);
    //                 atomicExch(node->mutex, 0);
    //                 locked = 0;
    //                 break;
    //             }
    //             if(node->body == -1){
    //                 printf("body %d found an empty leaf\n", body);
    //                 update_cuda(node, body, bx, by, bz, bmass);
    //                 decrement_semaphore();
    //                 printf("sem value: %d\n", *semaphore);
    //                 atomicExch(node->mutex, 0);
    //                 if(*semaphore == 0) break;
    //                 printf("node mutex: %d\n", *(node->mutex));
    //                 while(node->body >= 0 && *semaphore != 0){
    //                     // printf("body %d waiting.. sem value: %d\n", body, *semaphore);
    //                     continue;
    //                 }
    //                 if(*semaphore == 0) break;
    //                 increment_semaphore();
    //                 parent = node;
    //                 node = get_octant_cuda(parent, bx, by, bz);
    //                 locked = 0;
    //                 break;
    //             }
    //             if(node->body >= 0){
    //                 printf("body %d found a full leaf\n", body);
    //                 update_cuda(node, body, bx, by, bz, bmass);
    //                 generate_empty_children_cuda(node);
    //                 parent = node;
    //                 node = get_octant_cuda(parent, bx, by, bz);
    //                 atomicExch(node->mutex, 0);
    //                 locked = 0;
    //                 break;
    //             }
    //         }
    //     }
    //     printf("body %d DESAPARECIDO, sem value: %d\n", body, *semaphore);
    // }

}


__device__ void increment_semaphore(){
    int locked = 1;
    while(locked){
        if(atomicCAS(semaphore_mutex, 0, 1) == 0){
            (*semaphore)++;
            atomicExch(semaphore_mutex, 0);
            locked = 0;
        }
    }
}

__device__ void decrement_semaphore(){
    int locked = 1;
    while(locked){
        if(atomicCAS(semaphore_mutex, 0, 1) == 0){
            (*semaphore)--;
            atomicExch(semaphore_mutex, 0);
            locked = 0;
        }
    }
}
__device__ void update_cuda(bnode_cuda *node, int body, double body_x, double body_y, double body_z, double body_mass){
    if(node->body >= 0){
        node->body = -2;
    }
    if(node->body == -1){
        node->body = body;
    }
    double c_mass = node->mass + body_mass;
	double c_x = ((node->mass*node->x)+(body_mass*body_x))/c_mass;
	double c_y = ((node->mass*node->y)+(body_mass*body_y))/c_mass;
	double c_z = ((node->mass*node->z)+(body_mass*body_z))/c_mass;
	node->mass = c_mass;
	node->x = c_x;
	node->y = c_y;
	node->z = c_z;
}

// un vettore diviso n thread
__global__ void get_max_x_cuda(double *result){
    extern __shared__ double sdata_x[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    
    sdata_x[tid] = fabsf(x_d[i]);
    
    __syncthreads();

    for(unsigned int s=blockDim.x/2; s>0; s>>=1) {
	    if (tid < s) {
            if(fabsf(sdata_x[tid + s]) > fabsf(sdata_x[tid])){
                sdata_x[tid] = fabsf(sdata_x[tid + s]);
            }
        }
	    __syncthreads();
    }

    if(tid == 0) {
        *result = fabsf(sdata_x[0]);
    }
}

__global__ void get_max_y_cuda(double *result){
    extern __shared__ double sdata_y[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    
    sdata_y[tid] = fabsf(y_d[i]);
    
    __syncthreads();

    for(unsigned int s=blockDim.x/2; s>0; s>>=1) {
	    if (tid < s) {
            if(fabsf(sdata_y[tid + s]) > fabsf(sdata_y[tid])){
                sdata_y[tid] = fabsf(sdata_y[tid + s]);
            }
        }
	    __syncthreads();
    }

    if(tid == 0) {
        *result = fabsf(sdata_y[0]);
    }
}


__global__ void get_max_z_cuda(double *result){
    extern __shared__ double sdata_z[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    
    sdata_z[tid] = fabsf(z_d[i]);
    
    __syncthreads();

    for(unsigned int s=blockDim.x/2; s>0; s>>=1) {
	    if (tid < s) {
            if(fabsf(sdata_z[tid + s]) > fabsf(sdata_z[tid])){
                sdata_z[tid] = fabsf(sdata_z[tid + s]);
            }
        }
	    __syncthreads();
    }

    if(tid == 0) {
        *result = fabsf(sdata_z[0]);
    }
}

__device__ void generate_empty_children_cuda(bnode_cuda *node){
    int depth = node->depth+1;
	int scalar = fabsf(node->max_x - node->min_x)/2;
    bnode_cuda *o0, *o1, *o2, *o3, *o4, *o5, *o6, *o7;

    cudaMalloc((void**)&o0, sizeof(bnode_cuda));
    cudaMalloc((void**)&o1, sizeof(bnode_cuda));
    cudaMalloc((void**)&o2, sizeof(bnode_cuda));
    cudaMalloc((void**)&o3, sizeof(bnode_cuda));
    cudaMalloc((void**)&o4, sizeof(bnode_cuda));
    cudaMalloc((void**)&o5, sizeof(bnode_cuda));
    cudaMalloc((void**)&o6, sizeof(bnode_cuda));
    cudaMalloc((void**)&o7, sizeof(bnode_cuda));


    o0->depth = depth;
    o0->body = -1;
    cudaMalloc((void**)&o0->mutex, sizeof(int));
    *o0->mutex = 0;
    o0->min_x = node->min_x + scalar;
    o0->max_x = node->max_x;
    o0->min_y = node->min_y + scalar;
    o0->max_y = node->max_y;
    o0->min_z = node->min_z + scalar;
    o0->max_z = node->max_z;
	o0->x = 0;
	o0->y = 0;
	o0->z = 0;
	o0->mass = 0;

    o1->depth = depth;
    o1->body = -1;
    cudaMalloc((void**)&o1->mutex, sizeof(int));
    *o1->mutex = 0;
    o1->min_x = node->min_x;
    o1->max_x = node->max_x - scalar;
    o1->min_y = node->min_y + scalar;
    o1->max_y = node->max_y;
    o1->min_z = node->min_z + scalar;
    o1->max_z = node->max_z;
	o1->x = 0;
	o1->y = 0;
	o1->z = 0;
	o1->mass = 0;

    o2->depth = depth;
    o2->body = -1;
    cudaMalloc((void**)&o2->mutex, sizeof(int));
    *o2->mutex = 0;
    o2->min_x = node->min_x;
    o2->max_x = node->max_x - scalar;
    o2->min_y = node->min_y;
    o2->max_y = node->max_y - scalar;
    o2->min_z = node->min_z + scalar;
    o2->max_z = node->max_z;
	o2->x = 0;
	o2->y = 0;
	o2->z = 0;
	o2->mass = 0;

	o3->depth = depth;
	o3->body = -1;
    cudaMalloc((void**)&o3->mutex, sizeof(int));
    *o3->mutex = 0;
	o3->min_x = node->min_x + scalar;
	o3->max_x = node->max_x;
	o3->min_y = node->min_y;
	o3->max_y = node->max_y - scalar;
	o3->min_z = node->min_z + scalar;
	o3->max_z = node->max_z;
	o3->x = 0;
	o3->y = 0;
	o3->z = 0;
	o3->mass = 0;

	o4->depth = depth;
	o4->body = -1;
    cudaMalloc((void**)&o4->mutex, sizeof(int));
    *o4->mutex = 0;
	o4->min_x = node->min_x + scalar;
	o4->max_x = node->max_x;
	o4->min_y = node->min_y + scalar;
	o4->max_y = node->max_y;
	o4->min_z = node->min_z;
	o4->max_z = node->max_z - scalar;
	o4->x = 0;
	o4->y = 0;
	o4->z = 0;
	o4->mass = 0;

	o5->depth = depth;
	o5->body = -1;
    cudaMalloc((void**)&o5->mutex, sizeof(int));
    *o5->mutex = 0;
	o5->min_x = node->min_x;
	o5->max_x = node->max_x - scalar;
	o5->min_y = node->min_y + scalar;
	o5->max_y = node->max_y;
	o5->min_z = node->min_z;
	o5->max_z = node->max_z - scalar;
	o5->x = 0;
	o5->y = 0;
	o5->z = 0;
	o5->mass = 0;

	o6->depth = depth;
	o6->body = -1;
    cudaMalloc((void**)&o6->mutex, sizeof(int));
    *o6->mutex = 0;
	o6->min_x = node->min_x;
	o6->max_x = node->max_x - scalar;
	o6->min_y = node->min_y;
	o6->max_y = node->max_y - scalar;
	o6->min_z = node->min_z;
	o6->max_z = node->max_z - scalar;
	o6->x = 0;
	o6->y = 0;
	o6->z = 0;
	o6->mass = 0;

	o7->depth = depth;
	o7->body = -1;
    cudaMalloc((void**)&o7->mutex, sizeof(int));
    *o7->mutex = 0;
	o7->min_x = node->min_x + scalar;
	o7->max_x = node->max_x;
	o7->min_y = node->min_y;
	o7->max_y = node->max_y - scalar;
	o7->min_z = node->min_z;
	o7->max_z = node->max_z - scalar;
	o7->x = 0;
	o7->y = 0;
	o7->z = 0;
	o7->mass = 0;

    node->o0 = o0;
    node->o1 = o1;
    node->o2 = o2;
    node->o3 = o3;
    node->o4 = o4;
    node->o5 = o5;
    node->o6 = o6;
    node->o7 = o7;
}

__device__ bnode_cuda* get_octant_cuda(bnode_cuda* node, double x, double y, double z){
	int scalar = fabsf(node->max_x - node->min_x)/2;
    bnode_cuda* result;

	if(node->min_x + scalar <= x && x <= node->max_x && node->min_y + scalar <= y && y <= node->max_y && node->min_z + scalar <= z && z <= node->max_z){
        // printf("thread %d returning o0..\n", (blockIdx.x * blockDim.x) + threadIdx.x);
        result = node->o0;
    }
    if(node->min_x <= x && x <= node->max_x - scalar && node->min_y + scalar <= y && y <= node->max_y && node->min_z + scalar <= z && z <= node->max_z){
        // printf("thread %d returning o1..\n", (blockIdx.x * blockDim.x) + threadIdx.x);
        result = node->o1;
    }
    if(node->min_x <= x && x <= node->max_x - scalar && node->min_y <= y && y <= node->max_y - scalar && node->min_z + scalar <= z && z <= node->max_z){
        // printf("thread %d returning o2..\n", (blockIdx.x * blockDim.x) + threadIdx.x);
        result = node->o2;
    }
    if(node->min_x + scalar <= x && x <= node->max_x && node->min_y <= y && y <= node->max_y - scalar && node->min_z + scalar <= z && z <= node->max_z){
        // printf("thread %d returning o3..\n", (blockIdx.x * blockDim.x) + threadIdx.x);
        result = node->o3;
    }
    if(node->min_x + scalar <= x && x <= node->max_x && node->min_y + scalar <= y && y <= node->max_y && node->min_z <= z && z <= node->max_z - scalar){
        // printf("thread %d returning o4..\n", (blockIdx.x * blockDim.x) + threadIdx.x);
        result = node->o4;
    }
    if(node->min_x <= x && x <= node->max_x - scalar && node->min_y + scalar <= y && y <= node->max_y && node->min_z <= z && z <= node->max_z - scalar){
        // printf("thread %d returning o5..\n", (blockIdx.x * blockDim.x) + threadIdx.x);
        result = node->o5;
    }
    if(node->min_x <= x && x <= node->max_x - scalar && node->min_y <= y && y <= node->max_y - scalar && node->min_z <= z && z <= node->max_z - scalar){
        // printf("thread %d returning o6..\n", (blockIdx.x * blockDim.x) + threadIdx.x);
        result = node->o6;
    }
    if(node->min_x + scalar <= x && x <= node->max_x && node->min_y <= y && y <= node->max_y - scalar && node->min_z <= z && z <= node->max_z - scalar){
		// printf("thread %d returning o7..\n", (blockIdx.x * blockDim.x) + threadIdx.x);
        result = node->o7;
    }
    return result;
}

__device__ void print_node_cuda(bnode_cuda* node){
	printf("================================\nBODY: %d\nDEPTH: %d\nLOCK: %d\nMAX X: %d\nMAX Y: %d\nMAX Z: %d\nMIN X: %d\nMIN Y: %d\nMIN Z: %d\nX: %f\nY: %f\nZ: %f\nMASS: %f\n", node->body, node->depth, *node->mutex, node->max_x, node->max_y, node->max_z, node->min_x, node->min_y, node->min_z, node->x, node->y, node->z, node->mass);
}

// __global__ void build_barnes_tree_cuda(bnode* root){

// }
