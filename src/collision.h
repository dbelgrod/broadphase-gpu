#pragma once
#include <gpubf/aabb.h>

#define BLOCK_SIZE_1D 32 //sqrt(MAX_BLOCK_SIZE)
#define MAX_BLOCK_SIZE 1024 //for 1080Ti, V100

#define PADDING 1
#define BLOCK_PADDED BLOCK_SIZE_1D + PADDING

__global__ void reset_counter(int * counter);
__global__ void count_collisions(Aabb * boxes, int * count, int N);
__global__ void get_collision_pairs(Aabb * boxes, int * count, int * overlaps, int N, int G, const int nBoxesPerThread);
__global__ void get_collision_pairs_old(Aabb * boxes, int * count, int * overlaps, int N, int G);
