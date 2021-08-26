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

__device__ inline int lane_id(void) { return threadIdx.x % WARP_SIZE; }
// __device__ inline int warp_bcast(int mask, int v, int leader) { return __shfl_sync(mask, v, leader); }
// __device__ int atomicAggInc(int *ctr) {
// 		int mask = __ballot_sync(1), leader, res;
// 		// select the leader
// 		leader = __ffs(mask) - 1;
// 		// leader does the update
// 		if (lane_id() == leader)
// 			res = atomicAdd(ctr, __popc(mask));
// 		// broadcast result
// 		res = warp_bcast(mask, res, leader);
// 		// each thread computes its own value
// 		return res + __popc(mask & ((1 << lane_id()) - 1));
// 	}

// increment the value at ptr by 1 and return the old value
__device__ int atomicAggInc(int* ptr) {
    int mask;
#if __CUDA_ARCH__ >= 700
    mask = __match_any_sync(__activemask(), (unsigned long long)ptr);
#else
    unsigned tmask = __activemask();
    for (int i = 0; i < warpSize; i++){
#ifdef USE_OPT
      if ((1U<<i) & tmask){
#endif
        unsigned long long tptr = __shfl_sync(tmask, (unsigned long long)ptr, i);
        unsigned my_mask = __ballot_sync(tmask, (tptr == (unsigned long long)ptr));
        if (i == (threadIdx.x & (warpSize-1))) mask = my_mask;}
#ifdef USE_OPT
      }
#endif
#endif
    int leader = __ffs(mask) - 1;  // select a leader
    int res;
    unsigned lane_id = threadIdx.x % warpSize;
    if (lane_id == leader) {                 // leader does the update
        res = atomicAdd(ptr, __popc(mask));
    }
    res = __shfl_sync(mask, res, leader);    // get leader’s old value
    return res + __popc(mask & ((1 << lane_id) - 1)); //compute old value
}


__device__ void check_add_overlap(bool collides, int xid, int yid, int * count, int * overlaps, int G)
{
    if (collides)
        {
            int i = atomicAdd(count, 1);
            // int i = atomicAggInc(count);

            if (2*i + 1 < G)
            {
                overlaps[2*i] = xid;
                overlaps[2*i+1] = yid;
            }
        }

}

__global__ void get_collision_pairs(Aabb * boxes, int * count, int * overlaps, int N, int G, const int nBoxesPerThread)
{       
        extern __shared__ Aabb s_objects[];
        
        Aabb* s_x = &s_objects[0];
        Aabb* s_y = &s_objects[nBoxesPerThread*(BLOCK_PADDED)];


        int threadRowId = nBoxesPerThread*blockIdx.x * blockDim.x + threadIdx.x;
        int threadColId = nBoxesPerThread*blockIdx.y * blockDim.y + threadIdx.y;

        // printf("threadStart: %i %i\n", threadRowId, threadColId);
    
        // if (threadRowId >= N || threadColId >= N || threadColId >= threadRowId) return;
        // ex (threadRowId,threadColId) = (0,0) should not be considered but now it contains (1,0) so it must be incl.
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
        #pragma unroll 
        for (int i=0; i < nBoxesPerThread; i+=2)
        {
            #pragma unroll 
            for (int j=0; j < nBoxesPerThread; j+=2)
            // for (int j=nBoxesPerThread; j < 2* nBoxesPerThread; j++)
            {
                //reverse map to global mem
                int g_x__id = threadRowId + i*blockDim.x; 
                // int g_y__id = threadColId + (j-nBoxesPerThread)*blockDim.y; 
                int g_y__id = threadColId + j*blockDim.y; 

                // printf("threadTest: %i %i\n", g_x__id, g_y__id);

                if (g_x__id >= N || g_y__id >= N || g_y__id >= g_x__id) continue;

                const Aabb x = s_x[i*(BLOCK_PADDED) + threadIdx.x];      
                const Aabb y = s_y[j*(BLOCK_PADDED) + threadIdx.y];

                const Aabb w = s_x[(i+1)*(BLOCK_PADDED) + threadIdx.x];      
                const Aabb v = s_y[(j+1)*(BLOCK_PADDED) + threadIdx.y];

                // const Aabb a = s_x[(i+2)*(BLOCK_PADDED) + threadIdx.x];      
                // const Aabb b = s_y[(j+2)*(BLOCK_PADDED) + threadIdx.y];

                // const Aabb& x = s_x[(threadIdx.x+1)*nBoxesPerThread + i];      
                // const Aabb& y = s_y[(threadIdx.y+1)*nBoxesPerThread + j];
                
                
                bool collides = does_collide(x,y);
                bool collides2 = does_collide(x,v);
                bool collides3 = does_collide(w,v);
                bool collides4 = does_collide(w,y);

                // bool collides5 = does_collide(a,b);
                // bool collides6 = does_collide(a,v);
                // bool collides7 = does_collide(a,y);

                // bool collides8 = does_collide(x,b);
                // bool collides9 = does_collide(w,b);
                
                check_add_overlap(collides, g_x__id, g_y__id, count, overlaps, G);
                check_add_overlap(collides2, g_x__id, g_y__id, count, overlaps, G);
                check_add_overlap(collides3, g_x__id, g_y__id, count, overlaps, G);
                check_add_overlap(collides4, g_x__id, g_y__id, count, overlaps, G);
                // check_add_overlap(collides5, g_x__id, g_y__id, count, overlaps, G);
                // check_add_overlap(collides6, g_x__id, g_y__id, count, overlaps, G);
                // check_add_overlap(collides7, g_x__id, g_y__id, count, overlaps, G);
                // check_add_overlap(collides8, g_x__id, g_y__id, count, overlaps, G);
                // check_add_overlap(collides9, g_x__id, g_y__id, count, overlaps, G);
            }
        }
        // printf("\n");
    
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
        check_add_overlap(collides, threadRowId, threadColId, count, overlaps, G);
}
