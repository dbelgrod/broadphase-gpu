#pragma once

#include <gpubf/aabb.cuh>
#include <gpubf/collision.h>

__global__ struct SweepMarker {
    int id;
    float x,length;

    __device__ SweepMarker(int aabbId, Aabb& a)
    {
        id = aabbId;
        x = a.min.x;
        length = a.max.x - x;
    }

    __device__ SweepMarker() = default;
};

__global__ void build_index(Aabb * boxes, int N, int* index);
__global__ void print_sort_axis(Aabb* axis, int* index, int C);
__global__ void retrieve_collision_pairs(Aabb* boxes, int* index, int * count, int2 * overlaps, int N, int guess, int * numBoxes, int start = 0);