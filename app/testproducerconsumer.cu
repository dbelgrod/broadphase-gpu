#include <vector>
#include <iostream>
#include <bitset>
#include <string>
#include <numeric>
#include <string>
#include <functional>
#include <cuda/pipeline>
#include <cuda/semaphore>

// need this to get tiled_partition > 32 threads
#define _CG_ABI_EXPERIMENTAL // enable experimental API

#include <cooperative_groups.h>

#include <curand.h>
#include <curand_kernel.h>

#include <gpubf/queue.cuh>
#include <gpubf/aabb.cuh>
#include <gpubf/timer.cuh>
#include <gpubf/util.cuh>



using namespace std;
// using namespace cooperative_groups;
namespace cg = cooperative_groups;
typedef long long int ll;

__global__ void run(ll* in, ll * out, int N)
{
    __shared__ cuda::pipeline_shared_state<cuda::thread_scope_block, 2> pss;
    __shared__ Queue queue;
    queue.capacity = HEAP_SIZE;
    queue.heap_size = HEAP_SIZE;
    for (int i = threadIdx.x; i < HEAP_SIZE; i += blockDim.x) 
    {
        queue.lock[i].release();
        queue.harr[i].x = -1.0; //release to add
        // printf("Lock %i released\n", i);
    }
    __syncthreads();

    // Size must be a power of 2 and less than or equal to 32.
    cg::thread_block g = cg::this_thread_block();
    // thread_group tile32 = tiled_partition(g, 32);
    // int lane = tile32.thread_rank();
    // int tileIdx = g.thread_rank() / 32;
    
     // reserve shared memory for thread_block_tile usage.
    __shared__ cg::experimental::block_tile_memory<2, 1024> shared;
    cg::thread_block thb = cg::experimental::this_thread_block(shared);
 
    auto tilehalf = cg::experimental::tiled_partition<512>(thb);

    int lane = tilehalf.thread_rank();
    int lanerel = lane + blockIdx.x * blockDim.x;
    __syncthreads();

    int tid = threadIdx.x + blockDim.x*blockIdx.x;
    // if (tid >= N) return;
    // if in the first group + tid >= N
    if (tid >= N) return;

    // if (lane == 0)
    //     printf("group_rank: %i\n", tilehalf.meta_group_rank());
    
    if (tilehalf.meta_group_rank() == 0)
    {
        int curr1, curr2;
        int2 val1 = make_int2(in[tid],0);
        int2 val2 = make_int2(0, in[tid]);
        curr1 = queue.push(lanerel, val1);
        curr2 = queue.push(lanerel, val2);
    }
    else 
    {
        int2 res1 = queue.pop(lanerel  % HEAP_SIZE);
        // int2 res2 = queue.pop(lanerel % HEAP_SIZE);
        out[tid] = res1.x;
    }
    
    

//     int rand1, rand2;
//     /* CUDA's random number library uses curandState_t to keep track of the seed value
//      we will store a random state for every thread  */
//     curandState_t state;

//   /* we have to initialize the state */
//     curand_init(0, /* the seed controls the sequence of random values that are produced */
//                 0, /* the sequence number is only important with multiple cores */
//                 tid, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
//                 &state);

//   /* curand works like rand - except that it takes a state as a parameter */
//     rand1 = int(curand(&state) % HEAP_SIZE);
//     rand2 = int(curand(&state) % HEAP_SIZE);
    // if (tid == 0)
    //     printf("rand1: %i, rand2: %i\n", rand1, rand2);

    
    return;

}

int main( int argc, char **argv )
{
    vector<ll> nums;

    int N = atoi(argv[1]);


    for (ll i = 0; i < N; i++)
    {
        nums.push_back(i);
    }

    ll * d_in;
    cudaMalloc((void**)&d_in, sizeof(ll)*N);
    cudaMemcpy(d_in, nums.data(), sizeof(ll)*N, cudaMemcpyHostToDevice);

    ll * d_out;
    cudaMalloc((void**)&d_out, sizeof(ll)*N);
    cudaMemset(d_out, 0, sizeof(ll)*N);

    int block = 1024;
    int grid = (N / block + 1); 
    printf("grid size: %i\n", grid);
    printf("sizeof(semaphore):  %i\n", sizeof(cuda::binary_semaphore<cuda::thread_scope_block>));
    printf("sizeof(int2):  %i\n", sizeof(int2));

    recordLaunch("run", grid, block, 8, run, d_in, d_out, N);
    cudaDeviceSynchronize();

    vector<ll> out;
    out.resize(N);
    cudaMemcpy(out.data(), d_out, sizeof(ll)*N, cudaMemcpyDeviceToHost);

    // int s = accumulate(out.begin(), out.end(), 0);
    int s = 0;
    for (int i = 0; i < N; i+=1)
    {        
        // printf("%lld:%lld ", nums[i], out[i]);
        s += out[i];
    }
    printf("\n");
    printf("sum: %i\n", s);

}