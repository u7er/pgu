#include <cuda.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <stdlib.h>

__global__ void gTest(float* a, float* b, float* c){
    uint64_t i = threadIdx.x + blockDim.x * blockIdx.x;
    c[i] = b[i] + a[i];
    c[i] *= c[i];
}

int main(int argc, char *argv[]){
    float *da, *ha, *fa, *host1, *host2;
    uint64_t num_of_blocks=32000, threads_per_block=atoi(argv[1]);
    uint64_t N = 32000;
    num_of_blocks = N/threads_per_block;
    host1 = (float*)(calloc(N, sizeof(float)));
    host2 = (float*)(calloc(N, sizeof(float)));
    for (int i = 0; i < N; ++i) {
        host1[i] = i;
        host2[i] = i + 3;
    }

    cudaMalloc((void**)&da, N*sizeof(float));
    cudaMalloc((void**)&ha, N*sizeof(float));
    cudaMalloc((void**)&fa, N*sizeof(float));
    cudaMemcpy(da, host1, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(ha, host2, N*sizeof(float), cudaMemcpyHostToDevice);
    gTest<<<dim3((uint64_t)num_of_blocks),
    dim3((uint64_t)threads_per_block)>>>(da, ha, fa);
    cudaDeviceSynchronize();
    cudaMemcpy(host1, fa, N*sizeof(float), cudaMemcpyDeviceToHost);
    printf("%f", host1[1]);
    cudaFree(da);
    cudaFree(ha);
    cudaFree(fa);
    free(host1);
    free(host2);
    return 0;
}