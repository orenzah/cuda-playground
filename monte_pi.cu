#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>
__global__ void calculate(int seed, int *a , float* c, int N)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	curandState_t state;
	curand_init(seed + blockIdx.x, tid, 0, &state);
	//__shared__ int agg;
	//agg = agg + a[tid];
	int randX = curand(&state);
	float floatX = ((float)(randX % 1000000)) / 1000000;
	int randY = curand(&state);
	float floatY = ((float)(randY % 1000000)) / 1000000;
	if (powf(floatX,2) + powf(floatY,2) > 1)
	{
		a[tid] = 0;
	}
	else
	{
		a[tid] = 1;
	}
	//printf("tid: %d, random: %d\n", tid, rand % 25);
	__syncthreads();
	for (int j = 0; j < blockDim.x; j++)
	{
		int i = blockIdx.x * blockDim.x + j;
		c[blockIdx.x] = c[blockIdx.x] + ((float)a[i])/N;
	}
}
int main()
{
	float* gpu_x;
	int *gsum;
	int N = 1024;
	int B = 1 << 13;
	float* x = (float*) malloc(B * sizeof(float));

	cudaMalloc((void**) & gpu_x, B*sizeof(float));
	cudaMalloc((void**) & gsum, B*N*sizeof(int));
	cudaMemcpy(gsum, &sum, 10*sizeof(int), cudaMemcpyHostToDevice);
	calculate<<<B,N>>>(0, gsum, gpu_x, B*N);
	cudaMemcpy(x, gpu_x, B*sizeof(int), cudaMemcpyDeviceToHost);
	for (int i = 1; i < B; i++)
	{
		x[0] += x[i];
	}
	printf("Sum is %f\n", x[0]*4);
	cudaFree(gpu_x);
	cudaFree(gsum);
	free(x);
	return 0;
}
