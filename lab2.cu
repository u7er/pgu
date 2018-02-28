#include <cuda.h>
#include <driver_functions.h>
#include <driver_types.h>

#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>


#define CCHECK(ret) \
	if (_check_cuda((ret), __LINE__, __FILE__)) { \
		exit(EXIT_FAILURE); \
	}

int _check_cuda(cudaError_t err, int line, const char file[]) {
	if (err != cudaSuccess) {
		fprintf(stderr, "%s, %d in %s\n", cudaGetErrorString(err), line, file);
		return EXIT_FAILURE;
	}
	return EXIT_SUCCESS;
}

__global__ void Init(float *a) {
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	a[i] = i;
}

int main(int argc, char **argv) {
	float *da, *ha;
	int nob = atoi(argv[1]), tpb = 1024;
	int N = nob * tpb;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float elapsedTime = 0;

	ha = (float *) calloc(N, sizeof(float));
	CCHECK(cudaMalloc((void ** ) &da, N * sizeof(float)));

	cudaEventRecord(start, 0);
	Init<<<dim3(nob), dim3(tpb)>>>(da);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	//CCHECK(cudaDeviceSynchronize());

	CCHECK(cudaGetLastError());
	CCHECK(cudaMemcpy(ha, da, N * sizeof(float), cudaMemcpyDeviceToHost));
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("Elapsed time %gms\n", elapsedTime);

	for (int i = 0; i < N; ++i) {
		// printf("%d%c", ha[i], (i % 30 == 0 || i == (N - 1)? '\n' : ' '));
	}
	printf("End\n");
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	free(ha);
	CCHECK(cudaFree(da));
	return 0;
}
