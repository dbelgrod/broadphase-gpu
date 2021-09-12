#pragma once

#include <gpubf/aabb.h>
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

__global__ void build_sorting_axis(Aabb * boxes, int N, SweepMarker* axis);
__global__ void print_sort_axis(SweepMarker* axis, int C);
__global__ void retrieve_collision_pairs(SweepMarker* axis, Aabb* boxes, int * count, int2 * overlaps, int N, int guess);