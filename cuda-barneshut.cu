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

__device__ coord rres[] = {0, 0, 0, 0, 0, 0};

__global__ void ComputeBox(coord *x, coord *y, coord *z, coord *mass, coord *sx, coord *sy, coord *sz, int n, coord *result) {

        unsigned int tid = threadIdx.x;
        unsigned int i = blockIdx.x * blockDim.x + tid;

        extern __shared__ coord s_x[];
        coord *s_y = s_x + blockDim.x*2;
        coord *s_z = s_y + blockDim.x*2;

        s_x[tid] = 0;
        s_y[tid] = 0;
        s_z[tid] = 0;

        if (n <= i) {
                return;
        }

        s_x[tid*2] = x[i];
        s_x[tid*2 + 1] = x[i];
        s_y[tid*2] = y[i];
        s_y[tid*2 + 1] = y[i];
        s_z[tid*2] = z[i];
        s_z[tid*2 + 1] = z[i];

        int max_index = tid * 2;
        int min_index = tid * 2 + 1;

        __syncthreads();
        for(unsigned int s = 1; s < n; s *= 2) {
                if (tid % (2*s) == 0 && tid + s < n) {
                        s_x[tid*2] = fmax(s_x[tid*2], s_x[tid*2 + 2*s]);
                        s_x[tid*2 + 1] = fmin(s_x[tid*2 + 1], s_x[tid*2 + 2*s + 1]);
                        s_y[tid*2] = fmax(s_y[tid*2], s_y[tid*2 + 2*s]);
                        s_y[tid*2 + 1] = fmin(s_y[tid*2 + 1], s_y[tid*2 + 2*s + 1]);
                        s_z[tid*2] = fmax(s_z[tid*2], s_z[tid*2 + 2*s]);
                        s_z[tid*2 + 1] = fmin(s_z[tid*2 + 1], s_z[tid*2 + 2*s + 1]);

                }
                __syncthreads();
        }

        if (tid == 0) {
                result[0] = s_x[0];
                result[1] = s_x[1];
                result[2] = s_y[0];
                result[3] = s_y[1];
                result[4] = s_z[0];
                result[5] = s_z[1];
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

        ComputeBox<<<blocks, th_per_block, th_per_block*sizeof(coord)*3*2>>>(X, Y, Z, MASS, SX, SY, SZ, N, D_minmax);

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
