#include <gpubf/collision.h>

__global__ void count_collisions(Aabb * boxes, int * count, int N){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    
    const Aabb& a = boxes[tid];
    const Aabb& b = boxes[tid];

    bool collides = 
        a.max[0] >= b.min[0] && a.min[0] <= b.max[0] &&
        a.max[1] >= b.min[1] && a.min[1] <= b.max[1] &&
        a.max[2] >= b.min[2] && a.min[2] <= b.max[2];

    if (collides)
        atomicAdd(count, 1);
}

__device__ bool does_collide(const Aabb& x, const Aabb& y)
{
    return 
        x.max[0] >= y.min[0] && x.min[0] <= y.max[0] &&
            x.max[1] >= y.min[1] && x.min[1] <= y.max[1] &&
            x.max[2] >= y.min[2] && x.min[2] <= y.max[2];
}

__device__ void check_add_overlap(bool collides, const Aabb& x, const Aabb& y, int * count, int * overlaps, int G)
{
    if (collides)
        {
            int i = atomicAdd(count, 1);

            if (2*i + 1 < G)
            {
                overlaps[2*i] = x.id;
                overlaps[2*i+1] = y.id;
            }
        }

}

__global__ void get_collision_pairs(Aabb * boxes, int * count, int * overlaps, int N, int G, const int nBoxesPerThread)
{   
        extern __shared__ Aabb s_objects[];
    
        int threadRowId = nBoxesPerThread*blockIdx.x * blockDim.x + threadIdx.x;
        int threadColId = nBoxesPerThread*blockIdx.y * blockDim.y + threadIdx.y;

        if (threadRowId >= N || threadColId >= N || threadColId >= threadRowId) return;


        for (int shift = 0; shift < nBoxesPerThread; shift++)
        {
            int tidRow = threadRowId + shift*blockDim.x;
            int xIdx = (shift)*BLOCK_SIZE_1D + threadIdx.x;
            s_objects[xIdx] = boxes[tidRow];

            int tidCol = threadColId + shift*blockDim.y;
            int yIdx = (shift+nBoxesPerThread)*BLOCK_SIZE_1D + threadIdx.y;
            s_objects[yIdx] = boxes[tidCol];
        }

        for (int i=0; i < nBoxesPerThread; i++)
        {
            for (int j=nBoxesPerThread; j < 2*nBoxesPerThread; j++)
            {
                const Aabb& x = s_objects[i*BLOCK_SIZE_1D + threadIdx.x];      
                const Aabb& y = s_objects[j*BLOCK_SIZE_1D + threadIdx.y];
                
                bool collides = does_collide(x,y);
                check_add_overlap(collides, x, y, count, overlaps, G);
            }
        }
    
}

__global__ void reset_counter(int * counter){
    *counter = 0;
}

__global__ void get_collision_pairs_old(Aabb * boxes, int * count, int * overlaps, int N, int G)
{
    
        int threadRowId = blockIdx.x * blockDim.x + threadIdx.x;
        int threadColId = blockIdx.y * blockDim.y + threadIdx.y;
       
        if (threadRowId >= N || threadColId >= N || threadColId >= threadRowId) return;
    
        const Aabb& a = boxes[threadRowId];
        const Aabb& b = boxes[threadColId];
        bool collides = does_collide(a,b);
        check_add_overlap(collides, a, b, count, overlaps, G);
}
