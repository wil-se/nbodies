#include <cuda.h>
#include <stdio.h>
#include "device-common.h"
#include "cuda-barneshut.h"
#define G 6.67e-11
#define dt 1000000


__device__ int semaphore;

__global__ void compute_barneshut_forces_cuda(){
    int block_amount = (n_d/1024)+1;
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
    generate_empty_children_cuda(root);
    semaphore = n_d;
    printf("sem value: %d\n", semaphore);
    insert_body_cuda<<<block_amount, block_dim>>>(root);
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

__device__ bnode_cuda* get_octant(bnode_cuda* node, double x, double y, double z){
	int scalar = fabsf(node->max_x - node->min_x)/2;
    bnode_cuda* result;

	if(node->min_x + scalar <= x && x <= node->max_x && node->min_y + scalar <= y && y <= node->max_y && node->min_z + scalar <= z && z <= node->max_z){
        result = node->o0;
    }
    if(node->min_x <= x && x <= node->max_x - scalar && node->min_y + scalar <= y && y <= node->max_y && node->min_z + scalar <= z && z <= node->max_z){
        result = node->o1;
    }
    if(node->min_x <= x && x <= node->max_x - scalar && node->min_y <= y && y <= node->max_y - scalar && node->min_z + scalar <= z && z <= node->max_z){
        result = node->o2;
    }
    if(node->min_x + scalar <= x && x <= node->max_x && node->min_y <= y && y <= node->max_y - scalar && node->min_z + scalar <= z && z <= node->max_z){
        result = node->o3;
    }
    if(node->min_x + scalar <= x && x <= node->max_x && node->min_y + scalar <= y && y <= node->max_y && node->min_z <= z && z <= node->max_z - scalar){
        result = node->o4;
    }
    if(node->min_x <= x && x <= node->max_x - scalar && node->min_y + scalar <= y && y <= node->max_y && node->min_z <= z && z <= node->max_z - scalar){
        result = node->o5;
    }
    if(node->min_x <= x && x <= node->max_x - scalar && node->min_y <= y && y <= node->max_y - scalar && node->min_z <= z && z <= node->max_z - scalar){
        result = node->o6;
    }
    if(node->min_x + scalar <= x && x <= node->max_x && node->min_y <= y && y <= node->max_y - scalar && node->min_z <= z && z <= node->max_z - scalar){
		result = node->o7;
    }
    return result;
}

__global__ void insert_body_cuda(bnode_cuda *node){
    int body = (blockIdx.x * blockDim.x) + threadIdx.x;
    // bnode_cuda *parent = node;
    // double x = x_d[body], y = y_d[body], z = z_d[body], mass = mass_d[body];
    printf("HELLO FROM THREAD %d\n", body);
}
// __global__ void build_barnes_tree_cuda(bnode* root){

// }
