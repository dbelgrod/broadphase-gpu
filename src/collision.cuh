#pragma once

#include <gpubf/aabb.cuh>

#define BLOCK_SIZE_1D 32 // sqrt(MAX_BLOCK_SIZE)
// #define MAX_BLOCK_SIZE 1024 //for 1080Ti, V100
// #define WARP_SIZE 32

#define PADDING 0
#define BLOCK_PADDED BLOCK_SIZE_1D + PADDING

// template <typename T>
__global__ void reset_counter(uint *counter);
__global__ void reset_counter(int *counter);
__global__ void reset_counter(long long *counter);

__global__ void count_collisions(ccdgpu::Aabb *boxes, int *count, int N);
__global__ void get_collision_pairs(ccdgpu::Aabb *boxes, int *count,
                                    int2 *overlaps, int N, int G,
                                    const int nBoxesPerThread,
                                    long long *queries);
__global__ void get_collision_pairs_old(ccdgpu::Aabb *boxes, int *count,
                                        int2 *overlaps, int N, int G);

__device__ bool does_collide(const ccdgpu::Aabb &a, const ccdgpu::Aabb &b);
__device__ bool does_collide(ccdgpu::Aabb *a, ccdgpu::Aabb *b);
__device__ void add_overlap(const int &xid, const int &yid, int *count,
                            int2 *overlaps, int G);
__device__ bool covertex(const int3 &a, const int3 &b);
__device__ bool does_collide(const MiniBox &a, const MiniBox &b);

__device__ void append_queue(const int2 &lastcheck, int inc, int2 *queue,
                             unsigned *end);
