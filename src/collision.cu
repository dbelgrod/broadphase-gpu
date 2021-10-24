#include <gpubf/collision.h>

#include <cooperative_groups.h>
using namespace cooperative_groups;


__global__ void count_collisions(Aabb * boxes, int * count, int N){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    
    const Aabb& a = boxes[tid];
    const Aabb& b = boxes[tid];

    bool collides = 
        a.max.x >= b.min.x && a.min.x <= b.max.x &&
            a.max.y >= b.min.y && a.min.y <= b.max.y &&
            a.max.z >= b.min.z && a.min.z <= b.max.z;

    if (collides)
        atomicAdd(count, 1);
}

__device__ bool does_collide(const Aabb& a, const Aabb& b)
{
    return 
       a.max.x >= b.min.x && a.min.x <= b.max.x &&
            a.max.y >= b.min.y && a.min.y <= b.max.y &&
            a.max.z >= b.min.z && a.min.z <= b.max.z;
}

__device__ bool does_collide(Aabb* a, Aabb* b)
{
    return 
    //    a->max.x >= b->min.x && a->min.x <= b->max.x &&
            a->max.y >= b->min.y && a->min.y <= b->max.y &&
            a->max.z >= b->min.z && a->min.z <= b->max.z;
}

__device__ bool covertex(const int3& a, const int3& b) {
    
    return a.x == b.x || a.x == b.y || a.x == b.z || 
        a.y == b.x || a.y == b.y || a.y == b.z || 
        a.z == b.x || a.z == b.y || a.z == b.z;
}

// __device__ bool covertex(const float3& a, const float3& b) {
//     return a.x == b.x || a.x == b.y || a.x == b.z || 
//         a.y == b.x || a.y == b.y || a.y == b.z || 
//         a.z == b.x || a.z == b.y || a.z == b.z;
// }

// __device__ bool covertex_box(const Aabb a, const Aabb b) {
//     return a.max.x == b.max.x || a.max.x == b.max.y || a.max.x == b.max.z || 
//         a.max.y == b.max.x || a.max.y == b.max.y || a.max.y == b.max.z || 
//         a.max.z == b.max.x || a.max.z == b.max.y || a.max.z == b.max.z;
// }

__device__ void add_overlap(int& xid, int& yid, int * count, int2 * overlaps, int G)
{
    int i = atomicAdd(count, 1);

    if (i < G)
    {
        overlaps[i] = make_int2(xid, yid);
    } 
}

__global__ void get_collision_pairs(Aabb * boxes, int * count, int2 * overlaps, int N, int G, const int nBoxesPerThread, uint * queries)
{       
        extern __shared__ Aabb s_objects[];
        
        Aabb* s_x = &s_objects[0];
        Aabb* s_y = &s_objects[nBoxesPerThread*(BLOCK_PADDED)];


        int threadRowId = nBoxesPerThread*blockIdx.x * blockDim.x + threadIdx.x;
        int threadColId = nBoxesPerThread*blockIdx.y * blockDim.y + threadIdx.y;

        // ex (threadRowId,threadColId) = (0,0) should not be considered but now it contains (1,0) so it must be incl.
        
        //  atomicAdd(queries, 1);
        if (threadRowId >= N || threadColId >= N || threadColId - nBoxesPerThread*blockDim.y >= threadRowId) return;


        // #pragma unroll
        for (int shift = 0; shift < nBoxesPerThread; shift++)
        {
            int tidRow = threadRowId + shift*blockDim.x;
            int xIdx = (shift)*(BLOCK_PADDED) + threadIdx.x;
            // int xIdx = nBoxesPerThread*(threadIdx.x+1) + shift;
            s_x[xIdx]= boxes[tidRow];
            
        
            int tidCol = threadColId + shift*blockDim.y;
            int yIdx = (shift)*(BLOCK_PADDED) + threadIdx.y;
            // int yIdx = nBoxesPerThread*(threadIdx.y+1) + shift;
            s_y[yIdx] = boxes[tidCol];
        }

        // Aabb xboxes [30];
        // Aabb yboxes [30];
        // #pragma unroll 
        for (int i=0; i < nBoxesPerThread; i+=1)
        {
            // #pragma unroll 
            for (int j=0; j < nBoxesPerThread; j+=1)
            {
                //reverse map to global mem
                int g_x__id = threadRowId + i*blockDim.x; 
                // int g_y__id = threadColId + (j-nBoxesPerThread)*blockDim.y; 
                int g_y__id = threadColId + j*blockDim.y; 

                if (g_x__id >= N || g_y__id >= N || g_y__id >= g_x__id) continue;
               

                Aabb * x = &s_x[i*(BLOCK_PADDED) + threadIdx.x];      
                Aabb * y = &s_y[j*(BLOCK_PADDED) + threadIdx.y];

                // Aabb x = boxes[g_x__id];
                // Aabb y = boxes[g_y__id];
            

                if (
                    does_collide(x,y) //&&
                    // !covertex(xmax, ymax) &&
                    // !covertex(xmax, ymin) && 
                    // !covertex(xmin, ymin) &&
                    // !covertex(xmin, ymax)
                    )
                    add_overlap(g_x__id, g_y__id, count, overlaps, G);
            }
        }
    
}

// template<typename T>
__global__ void reset_counter(uint * counter){
    *counter = 0;
}

__global__ void reset_counter(int * counter){
    *counter = 0;
}

__global__ void get_collision_pairs_old(Aabb * boxes, int * count, int2 * overlaps, int N, int G)
{
    
        int threadRowId = blockIdx.x * blockDim.x + threadIdx.x;
        int threadColId = blockIdx.y * blockDim.y + threadIdx.y;
       
        if (threadRowId >= N || threadColId >= N || threadColId >= threadRowId) return;
    
        const Aabb& a = boxes[threadRowId];
        const Aabb& b = boxes[threadColId];
        if ( does_collide(a,b) )
            add_overlap(threadRowId, threadColId, count, overlaps, G);
}
