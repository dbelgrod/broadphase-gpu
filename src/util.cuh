#pragma once

#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

void setup(int devId, int& smemSize, int& threads, int& nbox)
{
    // Host code
    // int maxbytes = 98304; // 96 KB
    // cudaFuncSetAttribute(get_collision_pairs, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);

    int maxSmem;
    cudaDeviceGetAttribute(&maxSmem, 
        cudaDevAttrMaxSharedMemoryPerBlock, devId);
    printf("Max shared Memory per Block: %i B\n", maxSmem);

    int maxThreads;
    cudaDeviceGetAttribute(&maxThreads, 
        cudaDevAttrMaxThreadsPerBlock, devId);
    printf("Max threads per Block: %i thrds\n", maxThreads);

    nbox = nbox ? nbox : std::max((int)(smemSize / sizeof(Aabb) ) / maxThreads, 1);
    printf("Boxes per Thread: %i\n", nbox);

    

    // divide threads by an arbitrary number as long as its reasonable >64
    if (!threads)
    {
        cudaDeviceGetAttribute(&threads, 
            cudaDevAttrMaxThreadsPerMultiProcessor, devId);
        
        printf("Max threads per Multiprocessor: %i thrds\n", threads);
    }
    smemSize = nbox*threads*sizeof(Aabb);

    while (smemSize > maxSmem || threads > maxThreads)
    {
        threads--;
        smemSize = nbox*threads*sizeof(Aabb);
    }
    printf("Actual threads per Block: %i thrds\n", threads);
    printf("Shared mem alloc: %i B\n", smemSize);
    
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