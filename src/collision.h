#pragma once
#include <gpubf/aabb.h>


__global__ void reset_counter(int * counter);
__global__ void count_collisions(Aabb * boxes, int * count, int N);
__global__ void get_collision_pairs(Aabb * boxes, int * count, int * overlaps, int N);

