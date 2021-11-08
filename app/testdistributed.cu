// #include <gpubf/simulation.h>
#include <gpubf/simulation.h>
#include <gpubf/groundtruth.h>
#include <gpubf/util.cuh>


#include <thrust/sort.h>
#include <thrust/execution_policy.h>

#include <tbb/mutex.h>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/task_scheduler_init.h>
#include <tbb/enumerable_thread_specific.h>
#include "tbb/concurrent_vector.h"
#include <vector>
#include <iostream>
#include <vector>
#include <numeric>
#include <string>
#include <functional>

#include <cmath>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) 
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}


using namespace std;

__global__ void square_sum(int * d_in, int * d_out, int start, int end)
{
    int tid = start + threadIdx.x + blockIdx.x*blockDim.x;
    int gid = threadIdx.x + blockIdx.x*blockDim.x;
    
    if (tid >= end) return;
    d_out[gid] = d_in[tid] * d_in[tid];
}

void merge_local(
    const tbb::enumerable_thread_specific<vector<int>>& storages,
    std::vector<int>& overlaps)
{
    overlaps.clear();
    size_t num_overlaps = overlaps.size();
    for (const auto& local_overlaps : storages) {
        num_overlaps += local_overlaps.size();
    }
    // serial merge!
    overlaps.reserve(num_overlaps);
    for (const auto& local_overlaps : storages) {
        overlaps.insert(
            overlaps.end(), local_overlaps.begin(), local_overlaps.end());
    }
}


void run_sweep_multigpu(int N, int devcount)
{
    vector<int> squareSums;

    int in[N];
    for (int i = 0; i < N; i++)
        in[i] = N - i;

    cout<<"default threads "<<tbb::task_scheduler_init::default_num_threads()<<endl;
    // tbb::task_scheduler_init init(2);
    tbb::enumerable_thread_specific<vector<int>> storages;

    int device_init_id = 0;

    // int smemSize;
    // setup(device_init_id, smemSize, threads, nbox);

    cudaSetDevice(device_init_id);

    int * d_in;

    cudaMalloc((void**)&d_in, sizeof(int)*N);
    
    cudaMemcpy(d_in, in, sizeof(int)*N, cudaMemcpyHostToDevice);

    int threads = 1024;
    dim3 block(threads);
    int grid_dim_1d = (N / threads + 1); 
    dim3 grid( grid_dim_1d );

    try{
        thrust::sort(thrust::device, d_in, d_in + N);
        }
    catch (thrust::system_error &e){
        printf("Error: %s \n",e.what());}
    cudaDeviceSynchronize();
    

    int devices_count;
    cudaGetDeviceCount(&devices_count);
    // devices_count-=2;
    devices_count = devcount ? devcount : devices_count;
    int range = ceil( N / devices_count); 

    tbb::parallel_for(0, devices_count, 1, [&](int & device_id)    {

        
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device_id);
        printf("%s -> unifiedAddressing = %d\n", prop.name, prop.unifiedAddressing);

        cudaSetDevice(device_id);

        int is_able;

        for (int i=0; i<devices_count; i++)
        {
            cudaDeviceCanAccessPeer(&is_able, device_id, i);
            if (is_able)
            { 
                cudaDeviceEnablePeerAccess(i, 0);  
            }
            else if (i != device_id)
                printf("Device %i cant access Device %i\n", device_id, i);
        }
        

        gpuErrchk( cudaGetLastError() );   
    
        int range_start  = range*device_id;
        int range_end = range*(device_id + 1);
        printf("device_id: %i [%i, %i)\n", device_id, range_start, range_end);

        int * d_in_solo;
        cudaMalloc((void**)&d_in_solo, sizeof(int)*N);
        // if (device_id == device_init_id )
            cudaMemcpy(d_in_solo, d_in, sizeof(int)*N, cudaMemcpyDefault);
        
        // // turn off peer access for write variables
        sleep(1);
        for (int i=0; i<devices_count; i++)
        {
            cudaDeviceCanAccessPeer(&is_able, device_id, i);
            if (is_able)
            { 
                cudaDeviceDisablePeerAccess(i);  
            }
            else if (i != device_id)
                printf("Device %i cant access Device %i\n", device_id, i);
        }
        sleep(1);

        int * d_out;
        cudaMalloc((void**)&d_out, sizeof(int)*range);
        cudaMemset(d_out, 0, sizeof(int)*range);

    
        square_sum<<<grid, block>>>(d_in_solo, d_out, range_start, range_end);
        gpuErrchk(cudaDeviceSynchronize());

        int * out = (int*)malloc(sizeof(int)*range);
        gpuErrchk(cudaMemcpy(out, d_out, sizeof(int)*range, cudaMemcpyDeviceToHost));
       
        auto& local_overlaps = storages.local();
        
        for (size_t i=0; i < range; i++)
        {
           local_overlaps.emplace_back(out[i]);
        }
        
        printf("Total(filt.) overlaps for devid %i: %i\n", device_id, local_overlaps.size());
        // delete [] overlaps;
        // free(overlaps);
        
        // // free(counter);
        // // free(counter);
        // cudaFree(d_overlaps);
        // cudaFree(d_count); 
        // // cudaFree(d_b);
        // // cudaFree(d_r);
        // cudaDeviceReset();

    }); //end tbb for loop

    merge_local(storages, squareSums);

    int sum = accumulate(squareSums.begin(), squareSums.end(), 0);
    printf("\nFinal result: %i\n", sum);
    printf("Final result size: %i\n", squareSums.size());
    printf("\n");
    for (int i=0;i < N; i++)
    {
        printf("%i ", squareSums[i]);
    }
    printf("\n");

}

int main( int argc, char **argv )
{
    int N = 1;
    int devcount = 0;

    int o;
    while ((o = getopt (argc, argv, "n:d:")) != -1)
    {
        switch (o)
        {
            case 'n':
                N = atoi(optarg);
                break;
            case 'd':
                devcount = atoi(optarg);
                break;
        }
    }

    run_sweep_multigpu(N, devcount);

}
