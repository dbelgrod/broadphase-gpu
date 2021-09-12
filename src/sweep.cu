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

__global__ void print_sort_axis(Aabb* axis, int* index, int C)
{
    for (uint i = 0; i < C; i++)
        printf("id: %i, x: %.6f\n", index[i], axis[i].min.x);
}

__global__ void retrieve_collision_pairs(Aabb* boxes, int* index, int * count, int2 * overlaps, int N, int guess, int numBoxes)
{
    extern __shared__ Aabb s_objects[];

    int tid = numBoxes*blockIdx.x * blockDim.x + threadIdx.x;
    int ltid = threadIdx.x;

    if (tid >= N) return;

    for (int i=0; i < numBoxes; i++)
    {
        int t = tid + i*blockDim.x;
        int l = i*blockDim.x + ltid;
        s_objects[l]= boxes[t];
    }
    
    __syncthreads();
    

    for (int i=0; i < numBoxes; i++)
    {
        int t = tid + i*blockDim.x;
        int l = i*blockDim.x + ltid;
        
        if (t >= N) return;

        int ntid = t + 1;
        int nltid = l + 1;

        Aabb* a = &s_objects[l];
        Aabb* b = nltid < numBoxes*blockDim.x ? &s_objects[nltid] : &boxes[ntid];
        

        while (a->max.x  >= b->min.x)
        {
            if ( does_collide(*a,*b))
                add_overlap(index[t], index[ntid], count, overlaps, guess);
            
            ntid++;
            nltid++;
            if (ntid >= N) return;
            b = nltid < numBoxes*blockDim.x ? &s_objects[nltid] : &boxes[ntid];
        }
    } //for loop
}