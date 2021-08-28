// goal is to understand global/shared memory access and bank conflicts
#include <stdio.h>

__global__ void reverse(int* d_nums, int *d_rev, int N)
{
    extern __shared__ int s_nums[];
    int tid = threadIdx.x + blockIdx.x*blockDim.x;

    if (tid >= N) return;

    for (int i = 0; i < 1; i++)
    {
        s_nums[N-1-tid] = d_nums[tid];
    }

    for (int i = 0; i < 1; i++)
    {
        d_rev[tid] = s_nums[tid];
    }

}


int main()
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop); 

    const int N = 20;
    int nums[N];
    int rev[N];
        
    for (int i = 0; i<N; i++)
    {
        nums[i] = i;
        rev[i] = 0;
    }
    
    int * d_nums;
    int * d_rev; 
    cudaMalloc((void**)&d_nums, sizeof(int)*N);
    cudaMemcpy(d_nums, nums, sizeof(int)*N, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_rev, sizeof(int)*N);

    int ITER = 100;
    float milliseconds = 0;
    float avg_ms = 0;
    for (int i=0; i < ITER; i++)
    {
    cudaEventRecord(start);
    reverse<<<1,N, sizeof(int)*N>>>(d_nums, d_rev, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    avg_ms += milliseconds;
    }
    avg_ms /= ITER;
    printf("Avg. elapsed time: %.6f ms\n", avg_ms);

    cudaMemcpy(rev, d_rev, sizeof(int)*N, cudaMemcpyDeviceToHost);
    for (int i=0; i < N; i++)
        printf("%i ", rev[i]);
    printf("\n");
}

