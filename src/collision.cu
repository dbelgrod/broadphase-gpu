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

__global__ void get_collision_pairs(Aabb * boxes, int * count, int * overlaps, int N)
{
        // __shared__ Object s_objects[2*BLOCK_SIZE];
    
        int threadRowId = blockIdx.x * blockDim.x + threadIdx.x;
        int threadColId = blockIdx.y * blockDim.y + threadIdx.y;
       
        if (threadRowId >= N || threadColId >= N || threadColId <= threadRowId) return;
    
        // s_objects[threadIdx.x] = d_objects[threadRowId];
        // s_objects[BLOCK_SIZE + threadIdx.y] = d_objects[threadColId];
        // Aabb& a = s_objects[threadIdx.x].m_aabb;;
        // Aabb& b = s_objects[BLOCK_SIZE + threadIdx.y].m_aabb;
        const Aabb& a = boxes[threadRowId];
        const Aabb& b = boxes[threadColId];
    
        bool collides = 
            a.max[0] >= b.min[0] && a.min[0] <= b.max[0] &&
            a.max[1] >= b.min[1] && a.min[1] <= b.max[1] &&
            a.max[2] >= b.min[2] && a.min[2] <= b.max[2];
        
        if (collides)
        {
            int i = atomicAdd(count, 1);
            // d_overlaps[2*i] = s_objects[threadIdx.x].m_id;
            // d_overlaps[2*i+1] = s_objects[BLOCK_SIZE + threadIdx.y].m_id;
            overlaps[2*i] = a.id;
            overlaps[2*i+1] = b.id;
        }
        // __syncthreads();
    
}

__global__ void reset_counter(int * counter){
    *counter = 0;
}