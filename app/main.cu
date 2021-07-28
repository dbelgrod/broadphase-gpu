#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <igl/readOBJ.h>
#include <igl/edges.h>

#include <gpubf/aabb.h>

using namespace std;

__global__ void count_collisions(Aabb * boxes, int * count, int N){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    
    const Aabb& a = boxes[tid];
    const Aabb& b = boxes[tid];
    // Aabb b = boxes[tid];

    bool collides = 
        a.max[0] >= b.min[0] && a.min[0] <= b.max[0] &&
        a.max[1] >= b.min[1] && a.min[1] <= b.max[1] &&
        a.max[2] >= b.min[2] && a.min[2] <= b.max[2];

    if (collides)
        atomicAdd(count, 1);
}

__global__ void reset_counter(int * counter){
    *counter = 0;
}

int run_simulation(Aabb* boxes, int N) {
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
    return 1;
    // printf("%zu\n", sizeof(Aabb));


    // Retrieve count from GPU and print out
    // int counter;
    // cudaMemcpy(&counter, d_counter, sizeof(int), cudaMemcpyDeviceToHost);
    // printf("count: %d\n", counter);
    // return 0;
}

void constructBoxes
(
    Eigen::MatrixXd& vertices_t0, 
    Eigen::MatrixXd& vertices_t1, 
    Eigen::MatrixXi& faces, 
    Eigen::MatrixXi& edges, 
    vector<Aabb>& boxes
)
{
   addEdges(vertices_t0, vertices_t1, edges, boxes);
   addVertices(vertices_t0, vertices_t1, boxes);
   addFaces(vertices_t0, vertices_t1, faces, boxes);
}

void parseMesh(const char* filet0, const char* filet1, vector<Aabb>& boxes)
{

    // read in vertices, faces t=0
    Eigen::MatrixXd V0;
    Eigen::MatrixXi F;
    igl::readOBJ(filet0, V0, F);

    // get edges and close file
    Eigen::MatrixXi E;
    igl::edges(F,E);

    // read in vertices, t=1
    // faces should be same F^{t=0} = F^{t=1}
    Eigen::MatrixXd V1;    
    igl::readOBJ(filet1, V1, F);

    constructBoxes(V0, V1, F, E, boxes);
}


int main( int argc, const char* argv[] )
{
    const char* filet0 = argv[argc-2];
    const char* filet1 = argv[argc-1];
    
    vector<Aabb> boxes;
    parseMesh(filet0, filet1, boxes);

    Aabb* boxes_ptr = boxes.data();

    run_simulation(boxes_ptr, boxes.size());
}