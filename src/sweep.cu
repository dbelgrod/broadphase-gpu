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

__global__ void retrieve_collision_pairs(Aabb* boxes, int* index, int * count, int2 * overlaps, int N, int guess, int * numBoxes)
{
    extern __shared__ Aabb s_objects[];

    int tid = 1*blockIdx.x * blockDim.x + threadIdx.x;
    int ltid = threadIdx.x;

    if (tid >= N) return;

    #pragma unroll
    for (int i=0; i < numBoxes[0]; i++)
    {
        int t = tid + i*blockDim.x;
        int l = i*blockDim.x + ltid;
        s_objects[l]= boxes[t];
    }
    
    __syncthreads();
    


    int t = tid + 0*blockDim.x;
    int l = 0*blockDim.x + ltid;
    // tid = tid + 1*blockDim.x;
    // ltid = 1*blockDim.x + ltid;
    
    if (t >= N) return;

    int ntid = t + 1;
    int nltid = t + 1;

    Aabb* a = &s_objects[l];
    Aabb* b = nltid < numBoxes[0]*blockDim.x ? &s_objects[nltid] : &boxes[ntid];
    

    while (a->max.x  >= b->min.x) //boxes can touch and collide
    {
        if ( does_collide(a,b) 
            && !covertex(a->vertexIds, b->vertexIds)
            // && !covertex(a->min, b->max)
            // && !covertex(a->max, b->max)
            // && !covertex(a->max, b->min)
            )
            add_overlap(index[t], index[ntid], count, overlaps, guess);
        
        ntid++;
        nltid++;
        if (ntid >= N) return;
        b = nltid < numBoxes[0]*blockDim.x ? &s_objects[nltid] : &boxes[ntid];
    }
}