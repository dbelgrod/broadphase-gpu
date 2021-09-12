#include <gpubf/sweep.cuh>

__global__ void build_sorting_axis(Aabb * boxes, int N, SweepMarker* axis)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= N) return;

    Aabb a = boxes[tid];
    SweepMarker m = SweepMarker(tid, a);

    axis[tid] = m;
}

// __global__ void sort_sorting_axis(SweepMarker* axis)
// {

// }

__global__ void print_sort_axis(SweepMarker* axis, int C)
{
    for (uint i = 0; i < C; i++)
        printf("id: %i, x: %.6f, len: %.6f\n", axis[i].id, axis[i].x, axis[i].length);
}

__global__ void retrieve_collision_pairs(SweepMarker* axis, Aabb* boxes, int * count, int2 * overlaps, int N, int guess)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid > N) return;

    int i = 1;
    SweepMarker s = axis[tid];
    SweepMarker t = axis[tid + i];

    while (s.x + s.length >= t.x)
    {
        Aabb a = boxes[s.id];
        Aabb b = boxes[t.id];
        if ( does_collide(a,b))
            add_overlap(s.id, t.id, count, overlaps, guess);
        
        i++;
        if (tid + i > N) return;
        t = axis[tid + i];
    }
}