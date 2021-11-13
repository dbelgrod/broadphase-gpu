#include <gpubf/sweep.cuh>

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
    for (uint i = 0; i < C; i++)
        printf("id: %i, x: %.6f\n", axis[i].id, axis[i].min.x);
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

__global__ void create_sortedmin(Aabb * boxes, float3 * sortedmin, int N)
{

    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid >= N) return;

    sortedmin[tid] = make_float3(boxes[tid].min.x, boxes[tid].max.x, float(boxes[tid].id));
}

__global__ void build_checker(float3 * sortedmin, int2 * out, int N, int * count, int guess)
{
    // float3 x -> min, y -> max, z-> boxid
    extern __shared__ float3 s_sortedmin[];


    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid >= N) return;

    int ltid = threadIdx.x;

    s_sortedmin[ltid] = sortedmin[tid];
    
    __syncthreads();

    int ntid = tid + 1;
    int nltid = ltid + 1;

    if (ntid >= N) return;

    const float3& a = s_sortedmin[ltid];
    float3 b = nltid < blockDim.x ? s_sortedmin[nltid] : sortedmin[ntid];
    

    while (a.y  >= b.x) // curr max > following min
    {
        
        consider_pair(int(a.z), int(b.z), count, out, guess);
        
        ntid++;
        nltid++;
        if (ntid >= N) return;
        b = nltid < blockDim.x ? s_sortedmin[nltid] : sortedmin[ntid];
    }


}

__global__ void retrieve_collision_pairs2(const Aabb* const boxes, int * count, int2 * inpairs, int2 * overlaps, int N)
{
    extern __shared__ Aabb s_objects[];

    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    int ltid = threadIdx.x;

    if (tid >= N) return;

    s_objects[2*ltid] = boxes[inpairs[tid].x];
    s_objects[2*ltid+1] = boxes[inpairs[tid].y];

    const Aabb& a = s_objects[2*ltid];
    const Aabb& b = s_objects[2*ltid+1];
    
    if ( does_collide(a,b) 
            && !covertex(a.vertexIds, b.vertexIds)
    )
        add_overlap(a.id, b.id, count, overlaps, N);
    
}
