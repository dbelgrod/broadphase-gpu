#include <gpubf/sweep.cuh>
#include <cuda/pipeline>

__global__ void build_index(Aabb * boxes, int N, int* index)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= N) return;

    index[tid] = tid;
}

// __global__ void sort_sorting_axis(SweepMarker* axis)
// {

// }

__global__ void print_sort_axis(Aabb* axis, int C)
{
    // for (uint i = 0; i < C; i++)
    //     printf("id: %i, x: %.6f\n", axis[i].id, axis[i].min.x);
}

__global__ void print_overlap_start(int2 * overlaps)
{
    printf("overlap[0].x %d\n", overlaps[0].x);
}

__global__ void retrieve_collision_pairs(const Aabb* const boxes, int * count, int2 * overlaps, int N, int guess, int nbox, int start, int end)
{
    extern __shared__ Aabb s_objects[];

    // int gid = blockIdx.x * blockDim.x + threadIdx.x;
    // if (gid == 0)
    //     printf("index[0]-> %i\n", index[0]);

    int tid = start + threadIdx.x + blockIdx.x*blockDim.x;
    int ltid = threadIdx.x;

    if (tid >= N || tid >= end) return;

    // #pragma unroll
    // for (int i=0; i < nbox; i++)
    // {
    //     int t = tid + i*blockDim.x;
    //     int l = i*blockDim.x + ltid;
    //     s_objects[l]= boxes[t];
    // }
    s_objects[ltid] = boxes[tid];
    
    __syncthreads();
    


    int t = tid + 0*blockDim.x;
    int l = 0*blockDim.x + ltid;
    // tid = tid + 1*blockDim.x;
    // ltid = 1*blockDim.x + ltid;
    
    int ntid = t + 1;
    int nltid = l + 1;

    if (ntid >= N) return;

    const Aabb& a = s_objects[l];
    Aabb b = nltid < blockDim.x ? s_objects[nltid] : boxes[ntid];
    

    while (a.max.x  >= b.min.x) //boxes can touch and collide
    {
        if ( does_collide(a,b) 
            && !covertex(a.vertexIds, b.vertexIds)
        // if (tid % 100 == 0
        )
            // add_overlap(0, 0, count, overlaps, guess);
            add_overlap(a.id, b.id, count, overlaps, guess);
        
        ntid++;
        nltid++;
        if (ntid >= N) return;
        b = nltid < blockDim.x ? s_objects[nltid] : boxes[ntid];
    }
}


__device__ void consider_pair(const int& xid, const int& yid, int * count, int2 * out, int guess)
{
    int i = atomicAdd(count, 1);

    if (i < guess)
    {
        out[i] = make_int2(xid, yid);
    } 
}

// __global__ void create_sortedmin(Aabb * boxes, float3 * sortedmin, int N)
__global__ void create_ds(Aabb * boxes, float3 * sm, MiniBox * mini, int N, float3 * mean)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid >= N) return;

    sm[tid] = make_float3(boxes[tid].min.x, boxes[tid].max.x, float(tid));
    // sm[tid] = SortedMin(boxes[tid].min.x, boxes[tid].max.x, tid, boxes[tid].vertexIds);
    
    // float min[2] = {boxes[tid].min.y, boxes[tid].min.z};
    // float max[2] = {boxes[tid].max.y, boxes[tid].max.z};
    float vertices[4] = {boxes[tid].min.y, boxes[tid].min.z, boxes[tid].max.y, boxes[tid].max.z};
    mini[tid] = MiniBox(vertices, boxes[tid].vertexIds);

    // if (tid < 5)
    // {
    //     printf("mini %.6f\n", mini[tid].min.x);
    //     printf("sm %i\n", sm[tid].id);
    // }
}

// __global__ void build_checker(float3 * sortedmin, int2 * out, int N, int * count, int guess)
__global__ void build_checker(float3 * sm, int2 * out, int N, int * count, int guess)
{
    // float3 x -> min, y -> max, z-> boxid
    extern __shared__ float3 s_sortedmin[];
    // __shared__ cuda::barrier<cuda::thread_scope_block> barrier;
    int nbox = 1;

    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid >= N) return;

    int ltid = threadIdx.x;
    
    // init(&barrier, 1);
    cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();
    pipe.producer_acquire();
    cuda::memcpy_async(s_sortedmin + ltid, sm + tid, sizeof(*sm), pipe);
    pipe.producer_commit();
    pipe.consumer_wait();
    // pipeline_producer_commit(pipe, barrier);
    // barrier.arrive_and_wait();
    // pipe.consumer_release();


    // for (int i=0; i < nbox; i++)
    //     s_sortedmin[i*blockDim.x + ltid] = sm[tid+i*blockDim.x];

    __syncthreads();

    // for (int i=0; i<nbox;i++)
    // {
    int ntid = tid + 1;
    int nltid = ltid + 1;
    // int ntid = tid+i*blockDim.x + 1;
    // int nltid = i*blockDim.x + ltid + 1;

    if (ntid >= N) return;

    const float3& a = s_sortedmin[ltid];
    float3 b = nltid < nbox*blockDim.x ? s_sortedmin[nltid] : sm[ntid];
    // const float3& a = sortedmin[tid];
    // float3 b = sortedmin[ntid];

    while (a.y  >= b.x) // curr max > following min
    {
        //if(!covertex(a.vertexIds, b.vertexIds))
            consider_pair(int(a.z), int(b.z), count, out, guess);
        
        ntid++;
        nltid++;
        if (ntid >= N) return;
        b = nltid < nbox*blockDim.x ? s_sortedmin[nltid] : sm[ntid];
        // b = sortedmin[ntid];
    }
// }


}

// __global__ void retrieve_collision_pairs2(const Aabb* const boxes, int * count, int2 * inpairs, int2 * overlaps, int N, int guess)
__global__ void retrieve_collision_pairs2(const MiniBox* const mini, int * count, int2 * inpairs, int2 * overlaps, int N, int guess)
{
    extern __shared__ MiniBox s_mini[];

    int nbox = 1;

    int tid = threadIdx.x + nbox*blockIdx.x * blockDim.x;
    int ltid = threadIdx.x;

    if (tid >= N) return;

    for (int i =0; i < nbox; i++){
    int aid = inpairs[tid+i*blockDim.x].x;
    int bid = inpairs[tid+i*blockDim.x].y;

    // s_mini[2*(ltid+i*blockDim.x)] = mini[aid];
    // s_mini[2*(ltid+i*blockDim.x)+1] = mini[bid];

    // const MiniBox& a = s_mini[2*(ltid+i*blockDim.x)];
    // const MiniBox& b = s_mini[2*(ltid+i*blockDim.x)+1];

  
    const MiniBox& a = mini[aid];
    const MiniBox& b = mini[bid];
    
    if ( does_collide(a,b) 
            && !covertex(a.vertexIds, b.vertexIds)
    ){
        add_overlap(aid, bid, count, overlaps, guess);
    }
}
    
}

// #include <math.h>
//     __global__ void var(float *input, float *output, unsigned N, float mean){

//       unsigned idx=threadIdx.x+(blockDim.x*blockIdx.x);
//       if (idx < N) output[idx] = __powf(input[idx]-mean, 2);
//     }

//////////////
__global__ void create_rankbox(Aabb * boxes, RankBox * rankboxes, int N)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid >= N) return;

    rankboxes[tid].aabb = &boxes[tid];
}
    

__global__ void register_rank_x(RankBox * rankboxes, int N)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= N) return;
    rankboxes[tid].rank_x = tid;
}

__global__ void register_rank_y(RankBox * rankboxes, int N)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= N) return;
    rankboxes[tid].rank_y = tid;
}

__device__ ull cantor(ull x, ull y)
{
    return (x+y)/2 * (x+y+1) + y;
}

__global__ void assign_rank_c(RankBox * rankboxes, int N)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= N) return;
    rankboxes[tid].rank_c = cantor(rankboxes[tid].rank_x, rankboxes[tid].rank_y);
}

__global__ void build_checker2(const RankBox * const rankboxes, int2 * overlaps, int N, int * count, int guess)
{
    extern __shared__ RankBox s_rank[];

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int ltid = threadIdx.x;

    if (tid >= N) return;

    int ntid = tid + 1;
    int nltid = ltid + 1;

    if (ntid >= N) return;

    s_rank[ltid] = rankboxes[tid];

    __syncthreads();

    const RankBox a = s_rank[ltid];
    RankBox b = rankboxes[ntid]; //nltid < blockDim.x ? s_rank[nltid] : rankboxes[ntid];
  
    while (a.aabb->max.x  >= b.aabb->min.x || a.aabb->max.y  >= b.aabb->min.y) // curr max > following min
    {
        if(b.rank_x <= a.rank_x && b.rank_y <= a.rank_y &&
            // does_collide(a.aabb, b.aabb) &&
            //  a.aabb->max.z >= b.aabb->min.z && a.aabb->min.z <= b.aabb->max.z &&
            !covertex(a.aabb->vertexIds, b.aabb->vertexIds)
        )
        {
            
            add_overlap(a.aabb->id, b.aabb->id, count, overlaps, guess);
        }
        
        ntid++;
        nltid++;
        if (ntid >= N) return;
        b =  rankboxes[ntid]; //nltid < blockDim.x ? s_rank[nltid] : rankboxes[ntid];
    }
}

__global__ void print_stats(RankBox * rankboxes, int N)
{
    for (int i=N-50; i < N; i++)
    {
        RankBox & curr = rankboxes[i];
        printf("id: %i -> rank_x %llu rank_y %llu rank_c %llu\n", curr.aabb->id, curr.rank_x, curr.rank_y, curr.rank_c);
    }
    // rankboxes[tid].rank_c = cantor(rankboxes[tid].rank_x, rankboxes[tid].rank_y);
}