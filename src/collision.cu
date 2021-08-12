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

        // s_objects[threadIdx.x] = boxes[threadRowId];
        // s_objects[BLOCK_SIZE_1D + threadIdx.y] = boxes[threadColId];

        // Aabb compBoxes[] = {a, c, e, g, b, d, f, h};
        // Aabb compBoxes[2];
        for (int shift = 0; shift < nBoxesPerThread; shift++)
        {
            int tidRow = threadRowId + shift*blockDim.x;
            // int xIdx = (2*shift)*BLOCK_SIZE_1D + threadIdx.x;
            int xIdx = (shift)*BLOCK_SIZE_1D + threadIdx.x;
            s_objects[xIdx] = boxes[tidRow];
    

            int tidCol = threadColId + shift*blockDim.y;
            // int yIdx = (2*shift+1)*BLOCK_SIZE_1D + threadIdx.y;
            int yIdx = (shift+nBoxesPerThread)*BLOCK_SIZE_1D + threadIdx.y;
            s_objects[yIdx] = boxes[tidCol];

            
        }
        // int threadRowIdShift1 = threadRowId + blockDim.x;
        // int threadColIdShift1 = threadColId + blockDim.y;

        // int threadRowIdShift2 = threadRowId + 2*blockDim.x;
        // int threadColIdShift2 = threadColId + 2*blockDim.y;

        // int threadRowIdShift3 = threadRowId + 3*blockDim.x;
        // int threadColIdShift3 = threadColId + 3*blockDim.y;

        
        // s_objects[2*BLOCK_SIZE_1D + threadIdx.x] = boxes[threadRowIdShift1];
        // s_objects[3*BLOCK_SIZE_1D + threadIdx.y] = boxes[threadColIdShift1];
        
        // s_objects[4*BLOCK_SIZE_1D + threadIdx.x] = boxes[threadRowIdShift2];
        // s_objects[5*BLOCK_SIZE_1D + threadIdx.y] = boxes[threadColIdShift2];
        // s_objects[6*BLOCK_SIZE_1D + threadIdx.x] = boxes[threadRowIdShift3];
        // s_objects[7*BLOCK_SIZE_1D + threadIdx.y] = boxes[threadColIdShift3];

        // const Aabb& a = s_objects[threadIdx.x];
        // const Aabb& b = s_objects[BLOCK_SIZE_1D + threadIdx.y];
        // const Aabb& c = s_objects[2*BLOCK_SIZE_1D + threadIdx.x];
        // const Aabb& d = s_objects[3*BLOCK_SIZE_1D + threadIdx.y];
        
        // const Aabb& e = s_objects[4*BLOCK_SIZE_1D + threadIdx.x];
        // const Aabb& f = s_objects[5*BLOCK_SIZE_1D + threadIdx.y];
        // const Aabb& g = s_objects[6*BLOCK_SIZE_1D + threadIdx.x];
        // const Aabb& h = s_objects[7*BLOCK_SIZE_1D + threadIdx.y];

        // Aabb compBoxes[8] = {a, c, e, g, b, d, f, h};

        // const Aabb& a = boxes[threadRowId];
        // const Aabb& b = boxes[threadColId];
        

        for (int i=0; i < 2*nBoxesPerThread; i++)
        {
            for (int j=i+1; j < 2*nBoxesPerThread; j++)
            {
                int e = i < nBoxesPerThread ? threadIdx.x : threadIdx.y;
                const Aabb& x = s_objects[i*BLOCK_SIZE_1D + e];
                int f = j < nBoxesPerThread ? threadIdx.x : threadIdx.y;
                const Aabb& y = s_objects[j*BLOCK_SIZE_1D + f];
                
                // printf("x:%i, y%i\n", x.id, y.id);
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
