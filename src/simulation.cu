#include <gpubf/simulation.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>


void setup(int devId, int& smemSize, int& threads);



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

void run_scaling(const Aabb* boxes,  int N, int desiredBoxesPerThread, vector<unsigned long>& finOverlaps)
{

    int devId = 0;
    cudaSetDevice(devId);

    int smemSize;
    int threads;

    setup(devId, smemSize, threads);
    const int nBoxesPerThread = desiredBoxesPerThread ? desiredBoxesPerThread : smemSize / sizeof(Aabb) / (2*(BLOCK_PADDED));
    printf("Boxes per Thread: %i\n", nBoxesPerThread);

    finOverlaps.clear();
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop); 

    // guess overlaps size
    int guess = 0;

    // Allocate boxes to GPU 
    Aabb * d_boxes;
    cudaMalloc((void**)&d_boxes, sizeof(Aabb)*N);
    cudaMemcpy(d_boxes, boxes, sizeof(Aabb)*N, cudaMemcpyHostToDevice);

    // Allocate counter to GPU + set to 0 collisions
    int * d_count;
    cudaMalloc((void**)&d_count, sizeof(int));
    reset_counter<<<1,1>>>(d_count);
    cudaDeviceSynchronize();

    //Count collisions
    count_collisions<<<1,1>>>(d_boxes, d_count, N); 
    int count;
    cudaMemcpy(&count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
    reset_counter<<<1,1>>>(d_count);
    printf("Total collisions from counting: %i\n", count);



    int2 * d_overlaps;
    cudaMalloc((void**)&d_overlaps, sizeof(int2)*(guess));

    dim3 block(BLOCK_SIZE_1D,BLOCK_SIZE_1D);
    // dim3 grid ( (N+BLOCK_SIZE_1D)/BLOCK_SIZE_1D,  (N+BLOCK_SIZE_1D)/BLOCK_SIZE_1D );
    int grid_dim_1d = (N+BLOCK_SIZE_1D)/ BLOCK_SIZE_1D / nBoxesPerThread;
    dim3 grid( grid_dim_1d, grid_dim_1d );
    printf("Grid dim (1D): %i\n", grid_dim_1d);
    printf("Box size: %i\n", sizeof(Aabb));

    long long * d_queries;
    cudaMalloc((void**)&d_queries, sizeof(long long)*(1));
    reset_counter<<<1,1>>>(d_queries);

    printf("Shared mem alloc: %i B\n", nBoxesPerThread*2*(BLOCK_PADDED)*sizeof(Aabb) );
    cudaEventRecord(start);
    get_collision_pairs<<<grid, block, nBoxesPerThread*2*(BLOCK_PADDED)*sizeof(Aabb)>>>(d_boxes, d_count, d_overlaps, N, guess, nBoxesPerThread, d_queries);
    // get_collision_pairs_old<<<grid, block>>>(d_boxes, d_count, d_overlaps, N, guess);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    // cudaDeviceSynchronize();

    long long queries;
    cudaMemcpy(&queries, d_queries, sizeof(long long), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    printf("queries: %llu\n", queries);
    printf("needed queries: %llu\n", (long long)N*(N-1)/2 );

    // int count;
    cudaMemcpy(&count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    

    if (count > guess) //we went over
    {
        printf("Running again\n");
        cudaFree(d_overlaps);
        cudaMalloc((void**)&d_overlaps, sizeof(int2)*(count));
        reset_counter<<<1,1>>>(d_count);
        cudaDeviceSynchronize();
        cudaEventRecord(start);
        get_collision_pairs<<<grid, block, nBoxesPerThread*2*(BLOCK_PADDED)*sizeof(Aabb)>>>(d_boxes, d_count, d_overlaps, N, count, nBoxesPerThread, d_queries);
        // get_collision_pairs_old<<<grid, block>>>(d_boxes, d_count, d_overlaps, N, 2*count);
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
    // printf("Elapsed time: %.15f us/query\n", (milliseconds*1000)/((long long)N*N/2));

    int2 * overlaps =  (int2*)malloc(sizeof(int2) * (count));
    cudaMemcpy( overlaps, d_overlaps, sizeof(int2)*(count), cudaMemcpyDeviceToHost);


    cudaFree(d_overlaps);
    for (size_t i=0; i< count; i++)
    {
        // finOverlaps.push_back(overlaps[i].x);
        // finOverlaps.push_back(overlaps[i].y);
        
        const Aabb& a = boxes[overlaps[i].x];
        const Aabb& b = boxes[overlaps[i].y];
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
            finOverlaps.push_back(min(a.ref_id, b.ref_id));
            finOverlaps.push_back(max(a.ref_id, b.ref_id));
        }
    }

    printf("Total(filt.) overlaps: %lu\n", finOverlaps.size() / 2);
    free(overlaps);
    // free(counter);
    // free(counter);
    cudaFree(d_count);
    cudaDeviceReset();

}
//  // // //////// / / // / / // / // // / //  /

struct sort_sweepmarker_x
{
  __host__ __device__
  bool operator()(const SweepMarker &a, const SweepMarker &b) const {
    return (a.x < b.x);}
};

struct sort_aabb_x
{
  __host__ __device__
  bool operator()(const Aabb &a, const Aabb &b) const {
    return (a.min.x < b.min.x);}
};


void run_sweep(const Aabb* boxes, int N, int numBoxes, vector<unsigned long>& finOverlaps)
{
    int devId = 0;
    cudaSetDevice(devId);

    int smemSize;
    int maxBlockSize;

    setup(devId, smemSize, maxBlockSize );

    const int nBoxesPerThread = numBoxes ? numBoxes : std::max((int)(smemSize / sizeof(Aabb) ) / maxBlockSize,1);
    printf("Boxes per Thread: %i\n", nBoxesPerThread);
    printf("Shared mem alloc: %i B\n", nBoxesPerThread*maxBlockSize*sizeof(Aabb) );

    // Set shared memory per block to min of recommended and available
    smemSize = std::min(smemSize, (int)(nBoxesPerThread*maxBlockSize*sizeof(Aabb)) );
    printf("Actual shared mem alloc: %i B\n", smemSize);

    int * nbox;
    cudaMalloc((void**)&nbox, sizeof(int));
    cudaMemcpy(nbox, &nBoxesPerThread, sizeof(int), cudaMemcpyHostToDevice);

    finOverlaps.clear();
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop); 

    // Allocate boxes to GPU 
    Aabb * d_boxes;
    cudaMalloc((void**)&d_boxes, sizeof(Aabb)*N);
    cudaMemcpy(d_boxes, boxes, sizeof(Aabb)*N, cudaMemcpyHostToDevice);

    // Allocate counter to GPU + set to 0 collisions
    int * d_count;
    cudaMalloc((void**)&d_count, sizeof(int));
    reset_counter<<<1,1>>>(d_count);
    cudaDeviceSynchronize();


    // int SWEEP_BLOCK_SIZE = 1024;
    
    maxBlockSize = 512;
    dim3 block(maxBlockSize);
    int grid_dim_1d = (N / maxBlockSize + 1); 
    dim3 grid( grid_dim_1d );
    printf("Grid dim (1D): %i\n", grid_dim_1d);
    printf("Box size: %i\n", sizeof(Aabb));
    printf("SweepMarker size: %i\n", sizeof(SweepMarker));

    // int* d_index;
    // cudaMalloc((void**)&d_index, sizeof(int)*(N));
    int* rank;
    cudaMalloc((void**)&rank, sizeof(int)*(1*N));

    int* rank_x = &rank[0];
    // int* rank_y = &rank[N];
    // int* rank_z = &rank[2*N];

    // Translate boxes -> SweepMarkers
    cudaEventRecord(start);
    build_index<<<grid,block>>>(d_boxes, N, rank_x);
    // build_index<<<grid,block>>>(d_boxes, N, rank_y);
    // build_index<<<grid,block>>>(d_boxes, N, rank_z);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Elapsed time for build: %.6f ms\n", milliseconds);

    // Thrust sort (can be improved by sort_by_key)
    cudaEventRecord(start);
    // thrust::sort(thrust::device, d_axis, d_axis + N, sort_sweepmarker_x() );
    thrust::sort_by_key(thrust::device, d_boxes, d_boxes + N, rank_x, sort_aabb_x() );
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Elapsed time for sort: %.6f ms\n", milliseconds);

    // Test print some sorted output
    // print_sort_axis<<<1,1>>>(d_boxes,rank_x, 5);
    cudaDeviceSynchronize();

    // Find overlapping pairs
    int guess = 0;
    int2 * d_overlaps;
    cudaMalloc((void**)&d_overlaps, sizeof(int2)*(guess));

    int count;
    retrieve_collision_pairs<<<grid, block, smemSize>>>(d_boxes, rank_x, d_count, d_overlaps, N, guess, nbox);
    cudaMemcpy(&count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    if (count > guess) //we went over
    {
        printf("Running again\n");
        cudaFree(d_overlaps);
        cudaMalloc((void**)&d_overlaps, sizeof(int2)*(count));
        reset_counter<<<1,1>>>(d_count);
        cudaDeviceSynchronize();
        cudaEventRecord(start);
        retrieve_collision_pairs<<<grid, block, smemSize>>>(d_boxes, rank_x, d_count, d_overlaps, N, count, nbox);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        printf("Elapsed time for findoverlaps: %.6f ms\n", milliseconds);
    }
    // int count;
    cudaMemcpy(&count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    printf("Elapsed time: %.6f ms\n", milliseconds);
    printf("Collisions: %i\n", count);
    printf("Elapsed time: %.9f ms/collision\n", milliseconds/count);
    printf("Boxes: %i\n", N);
    printf("Elapsed time: %.9f ms/box\n", milliseconds/N);

    int2 * overlaps =  (int2*)malloc(sizeof(int2) * (count));
    cudaMemcpy( overlaps, d_overlaps, sizeof(int2)*(count), cudaMemcpyDeviceToHost);

    printf("Final count: %i\n", count);

    cudaFree(d_overlaps);
    for (size_t i=0; i< count; i++)
    {
        // finOverlaps.push_back(overlaps[i].x);
        // finOverlaps.push_back(overlaps[i].y);
        
        const Aabb& a = boxes[overlaps[i].x];
        const Aabb& b = boxes[overlaps[i].y];
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
            finOverlaps.push_back(min(a.ref_id, b.ref_id));
            finOverlaps.push_back(max(a.ref_id, b.ref_id));
        }
    }

    printf("Total(filt.) overlaps: %lu\n", finOverlaps.size() / 2);
    free(overlaps);
    // free(counter);
    // free(counter);
    cudaFree(d_count); 

    cudaDeviceReset();
}