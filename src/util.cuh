#pragma once

void setup(int devId, int& smemSize, int& threads)
{
    // Host code
    // int maxbytes = 98304; // 96 KB
    // cudaFuncSetAttribute(get_collision_pairs, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);

    // int smemSize;
    // int devId = 0;
    cudaDeviceGetAttribute(&smemSize, 
        cudaDevAttrMaxSharedMemoryPerBlock, devId);
    printf("Shared Memory per Block: %i B\n", smemSize);

    cudaDeviceGetAttribute(&threads, 
        cudaDevAttrMaxThreadsPerMultiProcessor, devId);
    printf("Max threads per Multiprocessor: %i thrds\n", threads);

    int maxThreads;
    cudaDeviceGetAttribute(&maxThreads, 
        cudaDevAttrMaxThreadsPerBlock, devId);
    printf("Max threads per Block: %i thrds\n", maxThreads);

    // divide threads by an arbitrary number as long as its reasonable >64
    int i = 2;
    int tmp = threads;
    while (tmp > maxThreads)
    {
        tmp = threads / i;
        i++;
    }
    threads = tmp;
    printf("Actual threads per Block: %i thrds\n", threads);
    
    // int warpSize;
    // cudaDeviceGetAttribute(&warpSize, 
    //     cudaDevAttrWarpSize, devId);
    // printf("Warp Size: %i\n", warpSize);
    
    // bank conflict avoid
    // cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);

    // cudaSharedMemConfig bankSize;
    // cudaDeviceGetSharedMemConfig(&bankSize);
    // printf("Bank size: %i\n", bankSize );
    

    return;
}