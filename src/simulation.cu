#include <gpubf/simulation.h>

// #define MAX_CONST_MEM 65536
// #define MAX_CONSTANT_BOXES MAX_CONST_MEM / sizeof(Aabb)

int setup_shared_memory()
{
    // Host code
    int maxbytes = 98304; // 96 KB
    cudaFuncSetAttribute(get_collision_pairs, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);

    int smemSize;
    int devId = 0;
    cudaDeviceGetAttribute(&smemSize, 
        cudaDevAttrMaxSharedMemoryPerBlockOptin, devId);
    printf("Shared Memory per Block: %i B\n", smemSize);

    return smemSize;
}


void run_collision_counter(Aabb* boxes, int N) {

    // int N = 200000;
    // Aabb boxes[N];
    // for (int i = 0; i<N; i++)
    // {
    //     boxes[i] = Aabb(i);
    //     // printf("box %i created\n", boxes[i].id);
    // }

    // Allocate boxes to GPU 
    Aabb * d_boxes;
    cudaMalloc((void**)&d_boxes, sizeof(Aabb)*N);
    cudaMemcpy(d_boxes, boxes, sizeof(Aabb)*N, cudaMemcpyHostToDevice);

    // Allocate counter to GPU + set to 0 collisions
    int * d_counter;
    cudaMalloc((void**)&d_counter, sizeof(int));
    reset_counter<<<1,1>>>(d_counter);
    cudaDeviceSynchronize();

     int collisions;
    // cudaMemcpy(&counter, d_counter, sizeof(int), cudaMemcpyDeviceToHost);

    // int bytes_mem_intrfce = 352 >> 3;
    // int mem_clock_rate = 1376 << 1;
    // float bandwidth_mem_theor = (mem_clock_rate * bytes_mem_intrfce) / pow(10, 3);

    // Set up timer
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Get number of collisions
    cudaEventRecord(start);
    count_collisions<<<1,1>>>(d_boxes, d_counter, N); 
    cudaEventRecord(stop);
    cudaMemcpy(&collisions, d_counter, sizeof(int), cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("(count_collisions<<<1,1>>>)\n");
    printf("Elapsed time: %.6f ms\n", milliseconds);
    printf("Elapsed time: %.6f ms/c\n", milliseconds/collisions);
    printf("Collision: %i\n", collisions);
    printf("Effective Bandwidth (GB/s): %.6f (GB/s)\n", 32*2/milliseconds/1e6);

    reset_counter<<<1,1>>>(d_counter);
    cudaEventRecord(start);
    count_collisions<<<1,1024>>>(d_boxes, d_counter, N); 
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaMemcpy(&collisions, d_counter, sizeof(int), cudaMemcpyDeviceToHost);
    printf("\n(count_collisions<<<1,1024>>>)\n");
    printf("Elapsed time: %.6f ms\n", milliseconds);
    printf("Elapsed time: %.6f ms/c\n", milliseconds/collisions);
    printf("Collision: %i\n", collisions);

    reset_counter<<<1,1>>>(d_counter);
    cudaEventRecord(start);
    count_collisions<<<2,1024>>>(d_boxes, d_counter, N); 
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaMemcpy(&collisions, d_counter, sizeof(int), cudaMemcpyDeviceToHost);
    printf("\n(count_collisions<<<2,1024>>>)\n");
    printf("Elapsed time: %.6f ms\n", milliseconds);
    printf("Elapsed time: %.6f ms/c\n", milliseconds/collisions);
    printf("Collision: %i\n", collisions);

    reset_counter<<<1,1>>>(d_counter);
    cudaEventRecord(start);
    count_collisions<<<56,1024>>>(d_boxes, d_counter, N); 
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaMemcpy(&collisions, d_counter, sizeof(int), cudaMemcpyDeviceToHost);
    printf("\n(count_collisions<<<56,1024>>>)\n");
    printf("Elapsed time: %.6f ms\n", milliseconds);
    printf("Elapsed time: %.9f ms/c\n", milliseconds/collisions);
    printf("Collision: %i\n", collisions);

    reset_counter<<<1,1>>>(d_counter);
    cudaEventRecord(start);
    count_collisions<<<256,1024>>>(d_boxes, d_counter, N); 
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaMemcpy(&collisions, d_counter, sizeof(int), cudaMemcpyDeviceToHost);
    printf("\n(count_collisions<<<256,1024>>>)\n");
    printf("Elapsed time: %.6f ms\n", milliseconds);
    printf("Elapsed time: %.9f ms/c\n", milliseconds/collisions);
    printf("Collision: %i\n", collisions);
    return;
    // printf("%zu\n", sizeof(Aabb));


    // Retrieve count from GPU and print out
    // int counter;
    // cudaMemcpy(&counter, d_counter, sizeof(int), cudaMemcpyDeviceToHost);
    // printf("count: %d\n", counter);
    // return 0;
}
// __constant__ Aabb d_boxes[MAX_CONSTANT_BOXES];
void run_scaling(const Aabb* boxes, int N, vector<unsigned long>& finOverlaps)
{
    int smemSize = setup_shared_memory();
    const int nBoxesPerThread = smemSize / sizeof(Aabb) / (2*BLOCK_SIZE_1D);
    printf("Boxes per Thread: %i\n", nBoxesPerThread);

    finOverlaps.clear();
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop); 

    // guess overlaps size
    int guess = 2*18*N;

    // Allocate boxes to GPU 
    Aabb * d_boxes;
    cudaMalloc((void**)&d_boxes, sizeof(Aabb)*N);
    cudaMemcpy(d_boxes, boxes, sizeof(Aabb)*N, cudaMemcpyHostToDevice);

    // Allocate counter to GPU + set to 0 collisions
    int * d_count;
    cudaMalloc((void**)&d_count, sizeof(int));
    reset_counter<<<1,1>>>(d_count);
    cudaDeviceSynchronize();

    int * d_overlaps;
    cudaMalloc((void**)&d_overlaps, sizeof(int)*(guess));

    dim3 block(BLOCK_SIZE_1D,BLOCK_SIZE_1D);
    // dim3 grid ( (N+BLOCK_SIZE_1D)/BLOCK_SIZE_1D,  (N+BLOCK_SIZE_1D)/BLOCK_SIZE_1D );
    int grid_dim_1d = (N+BLOCK_SIZE_1D)/BLOCK_SIZE_1D / nBoxesPerThread;
    dim3 grid( grid_dim_1d, grid_dim_1d );

    cudaEventRecord(start);
    get_collision_pairs<<<grid, block, nBoxesPerThread*2*BLOCK_SIZE_1D*sizeof(Aabb)>>>(d_boxes, d_count, d_overlaps, N, guess, nBoxesPerThread);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    // cudaDeviceSynchronize();

    int count;
    cudaMemcpy(&count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    if (2*count > guess) //we went over
    {
        printf("Running again\n");
        cudaFree(d_overlaps);
        cudaMalloc((void**)&d_overlaps, sizeof(int)*(count) * 2);
        reset_counter<<<1,1>>>(d_count);
        cudaDeviceSynchronize();
        cudaEventRecord(start);
        get_collision_pairs<<<grid, block, nBoxesPerThread*2*BLOCK_SIZE_1D*sizeof(Aabb)>>>(d_boxes, d_count, d_overlaps, N, 2*count, nBoxesPerThread);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        // cudaDeviceSynchronize();
    }
    printf("Elapsed time: %.6f ms\n", milliseconds);
    printf("Collisions: %i\n", count);
    printf("Elapsed time: %.9f ms/collision\n", milliseconds/count);
    printf("Boxes: %i\n", N);
    printf("Elapsed time: %.9f ms/box\n", milliseconds/N);

    int * overlaps =  (int*)malloc(sizeof(int) * (count)*2);
    cudaMemcpy( overlaps, d_overlaps, sizeof(int)*(count)*2, cudaMemcpyDeviceToHost);


    cudaFree(d_overlaps);
    for (size_t i=0; i< count; i++)
    {
        const Aabb& a = boxes[overlaps[2*i]];
        const Aabb& b = boxes[overlaps[2*i + 1]];
        if (a.type == Simplex::VERTEX && b.type == Simplex::FACE)
        {
            finOverlaps.push_back(a.ref_id);
            finOverlaps.push_back(b.ref_id);
        }
        else if (a.type == Simplex::FACE && b.type == Simplex::VERTEX)
        {
            finOverlaps.push_back(b.ref_id);
            finOverlaps.push_back(a.ref_id);
        }
        else if (a.type == Simplex::EDGE && b.type == Simplex::EDGE)
        {
            
            finOverlaps.push_back(b.ref_id);
            finOverlaps.push_back(a.ref_id);
        }
    }

    printf("Total(filt.) overlaps: %lu\n", finOverlaps.size() / 2);
    free(overlaps);
    // free(counter);
    // free(counter);
    cudaFree(d_count); 

}