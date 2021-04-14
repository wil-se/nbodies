

float4 *octree;
int *child_locks; 

// TODO isolare in .h esterno
#define GPU_CHECK(call) { checkCudaError(call, __FILE__, __LINE__); }

inline void checkCudaError( cudaError_t code, const char* s, int line) {
        if (code != cudaSuccess) {
                fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), s, line);
                printf("[CUDA ERROR} %s \n", cudaGetErrorString(code));
                printf("REASON: %s\n", cudaGetErrorName(code));
                exit(1);
        }

}

__host__ __device__ 
void init_octree(int nbodies) {
        size_t size = nbodies*8;
        GPU_CHECK(cudaMalloc((void**)&octree, sizeof(float4)*size));
        GPU_CHECK(cudaMalloc((void**)&child_locks, sizeof(int)*size));
}

__device__
int acquire_lock() {
        

}
