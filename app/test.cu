// goal is to understand global/shared memory access and bank conflicts
#include <stdio.h>

__global__ void reverse(int* d_nums, int *d_rev, int N)
{
    extern __shared__ int s_nums[];
    int g_tid = threadIdx.x + blockIdx.x*blockDim.x;
    int l_tid = threadIdx.x ;

    if (g_tid >= N) return;

    for (int i = 0; i < 1; i++)
    {
        s_nums[l_tid] = d_nums[g_tid];
    }
    
    for (int i = 0; i < 1; i++)
    {
        d_rev[N-g_tid-1] = s_nums[l_tid];
    }

}


int main(int argc, char **argv)
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop); 

    int N;
    if (argc == 2)
        N = atoi(argv[argc - 1]);
    else
        abort();

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

    int BLOCK_SIZE = 1024;
    int grid_size = N/BLOCK_SIZE + 1;
    printf("Grid size: %i\n", grid_size);

    int ITER = 1;
    float milliseconds = 0;
    float avg_ms = 0;
    for (int i=0; i < ITER; i++)
    {
    cudaEventRecord(start);
    reverse<<<grid_size, BLOCK_SIZE, 49152>>>(d_nums, d_rev, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    avg_ms += milliseconds;
    }
    avg_ms /= ITER;
    printf("Avg. elapsed time: %.6f ms\n", avg_ms);

    cudaMemcpy(rev, d_rev, sizeof(int)*N, cudaMemcpyDeviceToHost);
    long long sum = 0;
    for (int i=0; i < N; i++)
    {
        printf("%i ", rev[i]);
        sum+= rev[i];
    }
    // printf("\n");
    printf("sum -> %llu\n", sum);
}

