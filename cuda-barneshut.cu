#include <stdio.h>
#include <assert.h>
#include <cuda.h>
#include <string.h>

#define GPU_CHECK(call) { checkCudaError(call, __FILE__, __LINE__); }

inline void checkCudaError( cudaError_t code, const char* s, int line) {
        if (code != cudaSuccess) {
                fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), s, line);
                printf("[CUDA ERROR} %s \n", cudaGetErrorString(code));
                printf("REASON: %s\n", cudaGetErrorName(code));
                exit(1);
        }

}

#define ASYNC_CHECK() { checkCudaLastError(__FILE__, __LINE__); }

inline void checkCudaLastError(char* f, int l) {
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
                printf("Error at file %s, line %d\n", f, l);
                printf("[ERROR] %s\n", cudaGetErrorName(err)); 
                printf("[ERROR] %s\n", cudaGetErrorString(err)); 
        }
}

typedef double coord;


//__constant__ coord *X, *Y, *Z, *MASS, *SX, *SY, *SZ;


__device__ coord rres[] = {0, 0, 0, 0, 0, 0};

__global__ void ComputeBox(coord *x, coord *y, coord *z, coord *mass, coord *sx, coord *sy, coord *sz, int n, coord *result) {

        unsigned int tid = threadIdx.x;
        unsigned int i = blockIdx.x * blockDim.x + tid;

        extern __shared__ coord s_x[];

        if (n <= i) {
                return;
        }

        s_x[tid] = x[i];
        s_x[tid + 1] = y[i];
        s_x[tid + 2] = z[i];


        coord minmax[] = {s_x[tid], s_x[tid], s_x[tid + 1], s_x[tid + 1], s_x[tid + 2], s_x[tid + 2]};

        __syncthreads();
        for(unsigned int s = 3; s < n; s *= 2) {
                printf("[%d,%d] %lf, %lf, %lf\n\n", s, tid, s_x[tid*3], s_x[tid*3 + 1], s_x[tid*3 + 2]);
                if (tid % (2*s) == 0 && tid + s < n) {
                        printf("[%d] MAX ON X IS %f OVER %f AND %f\n", tid, fmax(minmax[0], s_x[tid + s]), minmax[0], s_x[tid + s]);
                        minmax[0] = fmax(minmax[0], s_x[tid + s]);
                        minmax[1] = fmin(minmax[1], s_x[tid + s]);
                        minmax[2] = fmax(minmax[2], s_x[tid + s + 1]);
                        minmax[3] = fmin(minmax[3], s_x[tid + s + 1]);
                        minmax[4] = fmax(minmax[4], s_x[tid + s + 2]);
                        minmax[5] = fmin(minmax[5], s_x[tid + s + 2]);

                }
                __syncthreads();
        }

        if (tid == 0) {
                for (int i = 0; i < 6; i++) {
                        result[i] = minmax[i];
                        printf("[%d] %lf\n ", i, minmax[i]);
                }
        }

}

void simulate_bh_cuda(){
        //cudaFuncSetCacheConfig(ComputeBox, cudaFuncCachePreferShared);
        int N;
        scanf("%d", &N);
        assert(N != 0);
        //TODO ridurre allocazione
        const long vec_size = N*sizeof(coord)*8;
        coord *x, *y, *z, *new_x, *new_y, *new_z, *new_sx, *new_sy, *new_sz, *mass, *sx, *sy, *sz;
        x = (coord*) malloc(vec_size);
        y = (coord*) malloc(vec_size);
        z = (coord*) malloc(vec_size);
        mass = (coord*) malloc(vec_size);
        sx = (coord*) malloc(vec_size);
        sy = (coord*) malloc(vec_size);
        sz = (coord*) malloc(vec_size);
        for (int i = 0; i < N; i++)
                scanf("%lf %lf %lf %lf %lf %lf %lf", &x[i], &y[i], &z[i], &mass[i], &sx[i], &sy[i], &sz[i]);

        coord *D_minmax;
        coord *minmax = (coord*) malloc(6*sizeof(coord));
        memset(minmax, 2, sizeof(coord)*6);
        coord *X, *Y, *Z, *MASS, *SX, *SY, *SZ;

        GPU_CHECK(cudaMalloc((void**)&X, vec_size));
        GPU_CHECK(cudaMalloc((void**)&Y, vec_size));
        GPU_CHECK(cudaMalloc((void**)&Z, vec_size));
        GPU_CHECK(cudaMalloc((void**)&MASS, vec_size));
        GPU_CHECK(cudaMalloc((void**)&SX, vec_size));
        GPU_CHECK(cudaMalloc((void**)&SY, vec_size));
        GPU_CHECK(cudaMalloc((void**)&SZ, vec_size));

        GPU_CHECK(cudaMalloc((void**)&D_minmax, sizeof(coord)*6));
        GPU_CHECK(cudaMemset(D_minmax, 0, sizeof(coord)*6));

        GPU_CHECK(cudaMemcpy(X, x, vec_size, cudaMemcpyHostToDevice));
        GPU_CHECK(cudaMemcpy(Y, y, vec_size, cudaMemcpyHostToDevice));
        GPU_CHECK(cudaMemcpy(Z, z, vec_size, cudaMemcpyHostToDevice));
        GPU_CHECK(cudaMemcpy(MASS, mass, vec_size, cudaMemcpyHostToDevice));
        GPU_CHECK(cudaMemcpy(SX, sx, vec_size, cudaMemcpyHostToDevice));
        GPU_CHECK(cudaMemcpy(SY, sy, vec_size, cudaMemcpyHostToDevice));
        GPU_CHECK(cudaMemcpy(SZ, sz, vec_size, cudaMemcpyHostToDevice));

        int th_per_block = 1024;
        int blocks = N/1024 + 1;

        ComputeBox<<<blocks, th_per_block, th_per_block*sizeof(coord)*3>>>(X, Y, Z, MASS, SX, SY, SZ, N, D_minmax);

        ASYNC_CHECK();

        cudaMemcpy(minmax, D_minmax, sizeof(coord)*6, cudaMemcpyDeviceToHost); 

        printf("%lf\t", minmax[0]); 
        printf("%lf\t", minmax[1]); 
        printf("%lf\t", minmax[2]); 
        printf("%lf\t", minmax[3]); 
        printf("%lf\t", minmax[4]); 
        printf("%lf\t", minmax[5]); 
        printf("\n");

}

int main() {
        simulate_bh_cuda();
}
