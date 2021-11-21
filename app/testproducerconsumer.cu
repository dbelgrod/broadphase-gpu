#include <vector>
#include <iostream>
#include <bitset>
#include <string>
#include <cuda/pipeline>
#include <cuda/semaphore>
#include <cooperative_groups.h>

#include <gpubf/queue.cuh>
#include <gpubf/aabb.cuh>

using namespace std;
typedef long long int ll;

__global__ void run(ll* in, ll * out, int N)
{
    __shared__ cuda::pipeline_shared_state<cuda::thread_scope_block, 1> pss;
    __shared__ Queue queue;
    queue.capacity = HEAP_SIZE;

    // extern __shared__ T s[];
    auto group = cooperative_groups::this_thread_block();
    // T* shared[2] = { s, s + 2 * group.size() };

      // Create a partitioned block-scoped pipeline where half the threads are producers.
    cuda::std::size_t producer_count = group.size() / 2;
    cuda::pipeline<cuda::thread_scope_block> pipe = cuda::make_pipeline(group, &pss, producer_count);

    // cuda::std::ptrdiff_t max = 1;
    extern __shared__ cuda::binary_semaphore<cuda::thread_scope_block> a[];
    cuda::binary_semaphore<cuda::thread_scope_block>* b[1] = {a};
    // a = cuda::binary_semaphore<cuda::thread_scope_block>(1);
    a[0].release();
    __syncthreads();

    int tid = threadIdx.x + blockDim.x*blockIdx.x;
    if (tid >= N) return;

    // Prime the pipeline.
    // pipe.producer_acquire();
    int2 val = make_int2(in[tid], in[tid]);
    if (a[0].try_acquire())
    {
        printf("tid %i acquired semaphore\n", tid);
        queue.push(val);
    }
    else
        printf("tid %i failed to acquire semaphore\n", tid);
    

    // pipe.producer_commit();

    // cuda::pipeline_consumer_wait_prior<1>(pipe);
    // pipe.consumer_wait();
    // // while (queue.size())
    // int2 res = queue.pop();
    // out[tid] = val.x * val.y;
    // pipe.consumer_release();
    // // Create a pipeline.

    // out[tid] = // atomicAdd(&var[0].x, __powf(boxes[tid].min.x-mean[0].x, 2));
    // out[tid] = __mulhi(f1,f2);
    
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

    run<<<grid, block, 80>>>(d_in, d_out, N);
    cudaDeviceSynchronize();

    vector<ll> out;
    out.reserve(N);
    cudaMemcpy(out.data(), d_out, sizeof(ll)*N, cudaMemcpyDeviceToHost);


    for (ll i = 0; i < N; i++)
    {        
        printf("%lld:%lld ", nums[i], out[i]);
    }
    printf("\n");

}